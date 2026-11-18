import json

import torch

from training.rollback_model import rollback_model


def _write_checkpoint(directory, bias: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 2),
    )
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.fill_(bias)
    traced = torch.jit.trace(model.eval(), torch.zeros(1, 3, 32, 32))
    traced.save(str(directory / "classifier.pt"))
    (directory / "class_mapping.json").write_text(json.dumps(["healthy", "disease"]), encoding="utf-8")
    (directory / "classifier_metadata.json").write_text(
        json.dumps({"classes": ["healthy", "disease"], "image_size": 32}),
        encoding="utf-8",
    )


def test_rollback_requires_confirmation_and_preserves_previous_model(tmp_path) -> None:
    production = tmp_path / "saved_models"
    backup = production / "baseline_backup_previous"
    _write_checkpoint(production, 1.0)
    _write_checkpoint(backup, 2.0)

    try:
        rollback_model(backup, production)
    except ValueError as exc:
        assert "confirmation" in str(exc)
    else:
        raise AssertionError("Rollback should require explicit confirmation.")

    previous_backup = rollback_model(backup, production, confirm=True)
    restored = torch.jit.load(str(production / "classifier.pt"), map_location="cpu")
    previous = torch.jit.load(str(previous_backup / "classifier.pt"), map_location="cpu")
    sample = torch.zeros(1, 3, 32, 32)

    assert torch.equal(restored(sample), torch.full((1, 2), 2.0))
    assert torch.equal(previous(sample), torch.full((1, 2), 1.0))