from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_rf(X_train, y_train):
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 12],
        "min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_