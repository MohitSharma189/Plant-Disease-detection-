from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Plant Disease Detection"
    environment: str = "development"
    cpu_threads: int = 2
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'plant_disease.db'}"
    model_dir: Path = PROJECT_ROOT / "saved_models"
    # Phase 6A.2.1: Deployed checkpoint configured to verified Phase 5 benchmark TorchScript model
    classifier_path: Path = (
        PROJECT_ROOT
        / "saved_models"
        / "experiments"
        / "mobilenet_v3_small_final"
        / "classifier.pt"
    )
    classifier_metadata_path: Path = (
        PROJECT_ROOT
        / "saved_models"
        / "experiments"
        / "mobilenet_v3_small_final"
        / "classifier_metadata.json"
    )
    class_mapping_path: Path = PROJECT_ROOT / "saved_models" / "class_mapping.json"
    calibration_enabled: bool = False  # Keep calibration disabled at T=1.0 until calibration discrepancy is independently resolved
    default_review_threshold: float = 0.60  # Operational engineering heuristic threshold
    report_dir: Path = PROJECT_ROOT / "reports" / "generated"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chat_api_url: str = ""
    chat_api_key: SecretStr | None = None
    chat_model: str = ""
    chat_timeout_seconds: float = 45.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PLANT_", extra="ignore")


settings = Settings()
