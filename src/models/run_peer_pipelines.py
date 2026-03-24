from pathlib import Path

from src.models.decision_tree_pipeline import run_decision_tree_pipeline
from src.models.svm_pipeline import run_svm_pipeline
from src.models.mlp_pipeline import run_mlp_pipeline


def run_peer_pipelines(df, report_root: Path):
    """Orquestar los tres pipelines (Árbol, SVM, Red Neuronal)
    Retorna lista de resultados con métricas de cada modelo"""
    
    # Ejecutar los tres pipelines - cada uno en su carpeta de algoritmo
    tree_result = run_decision_tree_pipeline(df, report_root / "arbol_decision")
    svm_result = run_svm_pipeline(df, report_root / "svm")
    mlp_result = run_mlp_pipeline(df, report_root / "red_neuronal")
    
    return [tree_result, svm_result, mlp_result]
