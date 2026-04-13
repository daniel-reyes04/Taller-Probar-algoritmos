import joblib
from pathlib import Path


def save_model(model, model_name: str, model_dir: Path = None) -> Path:
    if model_dir is None:
        model_dir = Path("models")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, file_path)
    return file_path


def load_model(model_path: Path):
    return joblib.load(model_path)
