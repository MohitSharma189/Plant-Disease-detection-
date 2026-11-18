import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

from training.train_definitive import (
    RandomMotionBlur,
    RandomShadow,
    build_fine_tune_scheduler,
    build_model,
    set_frozen_stages,
    save_atomic,
    verify_inference,
)


def test_backbone_frozen_and_unfrozen():
    model = nn.Module()
    model.features = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model.classifier = nn.Linear(4, 2)

    set_frozen_stages(model, train_backbone=False)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    set_frozen_stages(model, train_backbone=True)
    assert all(p.requires_grad for p in model.features.parameters())


def test_finetune_scheduler():
    param = nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = build_fine_tune_scheduler(opt, steps_per_epoch=2, epochs=3, warmup_epochs=1)
    assert opt.param_groups[0]["lr"] == 0.5
    for _ in range(6):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < 0.01


def test_field_augmentations():
    img = Image.new("RGB", (48, 32), "green")
    assert RandomMotionBlur(1.0)(img).size == img.size
    assert RandomShadow(1.0)(img).size == img.size


def test_model_build_and_dropout():
    model = build_model(class_count=3, head_dropout=0.3)
    assert model.classifier[-2].p == 0.3
    assert model.classifier[-1].out_features == 3


def test_save_atomic():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "sub" / "model.pt"
        data = {"epoch": 1, "tensor": torch.tensor([1, 2, 3])}
        save_atomic(data, target)
        assert target.is_file()
        loaded = torch.load(target)
        assert loaded["epoch"] == 1
        assert torch.equal(loaded["tensor"], torch.tensor([1, 2, 3]))


def test_checkpoint_contains_all_resumption_states():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "latest_checkpoint.pth"
        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        checkpoint = {
            "epoch": 2,
            "step": 500,
            "running_loss": 12.34,
            "running_correct": 420,
            "running_total": 500,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "best_validation_loss": 0.45,
            "best_validation_accuracy": 0.91,
            "best_validation_macro_f1": 0.89,
            "best_epoch": 1,
            "stale_epochs": 0,
            "history": [{"epoch": 1, "val_loss": 0.45}],
        }
        save_atomic(checkpoint, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["epoch"] == 2
        assert loaded["step"] == 500
        assert "optimizer_state_dict" in loaded
        assert "scheduler_state_dict" in loaded
        assert "torch_rng_state" in loaded
        assert "numpy_rng_state" in loaded
        assert "python_rng_state" in loaded
        assert loaded["running_loss"] == 12.34
