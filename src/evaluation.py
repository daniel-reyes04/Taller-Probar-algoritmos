from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_multiclass(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    # Convencion estandar: filas=real, columnas=predicho.
    cm_true_pred = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "exactitud": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "sensibilidad_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    return metrics, cm_true_pred