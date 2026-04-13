from pathlib import Path

import joblib

from src.models.decision_tree_pipeline import run_decision_tree_pipeline
from src.models.svm_pipeline import run_svm_pipeline
from src.models.mlp_pipeline import run_mlp_pipeline


def run_peer_pipelines(df, report_root: Path, model_root: Path = None):
    """Orquestar los tres pipelines (Árbol, SVM, Red Neuronal)
    Retorna lista de resultados con métricas de cada modelo"""

    if model_root is None:
        model_root = Path("models")
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)

    tree_result = run_decision_tree_pipeline(
        df, report_root / "arbol_decision", model_root
    )
    svm_result = run_svm_pipeline(df, report_root / "svm", model_root)
    mlp_result = run_mlp_pipeline(df, report_root / "red_neuronal", model_root)

    return [tree_result, svm_result, mlp_result]
