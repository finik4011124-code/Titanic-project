from sklearn.model_selection import cross_validate


def evaluate_cv(model, X, y, cv) -> dict:
   
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
    )
    return {
        "accuracy": results["test_accuracy"].mean(),
        "precision": results["test_precision"].mean(),
        "recall": results["test_recall"].mean(),
        "f1": results["test_f1"].mean(),
    }