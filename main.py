import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data, compute_fill_stats, preprocess, align_columns
from src.features import add_features, select_model_columns
from src.models import build_voting_classifier
from src.train import evaluate_cv


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    random_state = config["cv"]["random_state"]

    train_raw = load_data(config["data"]["train_path"])
    test_raw = load_data(config["data"]["test_path"])

 
    stats = compute_fill_stats(train_raw)
    train = preprocess(train_raw, stats)
    test = preprocess(test_raw, stats)

    train = add_features(train)
    test = add_features(test)


    y = train["Survived"]
    X = select_model_columns(train)
    X_test = align_columns(select_model_columns(test), reference_columns=X.columns)

 
    skf = StratifiedKFold(
        n_splits=config["cv"]["n_splits"], shuffle=True, random_state=random_state
    )


    voting = build_voting_classifier(
        X, y, skf, search_config=config["search"], random_state=random_state
    )

   
    metrics = evaluate_cv(voting, X, y, skf)
    print(
        f"Voting ensemble CV results — accuracy: {metrics['accuracy']:.3f}, "
        f"precision: {metrics['precision']:.3f}, "
        f"recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}"
    )

    
    voting.fit(X, y)
    predictions = voting.predict(X_test)

    
    submission = pd.DataFrame({
        "PassengerId": test_raw["PassengerId"],
        "Survived": predictions.astype(int),
    })
    submission.to_csv(config["data"]["output_path"], index=False)
    print(f"Saved submission to {config['data']['output_path']}")


if __name__ == "__main__":
    main()