import optuna
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _search_lightgbm(X, y, cv, n_trials, random_state):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "random_state": random_state,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    return LGBMClassifier(**study.best_params, random_state=random_state, verbose=-1)


def _search_catboost(X, y, cv, n_trials, random_state):
    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "verbose": 0,
            "random_state": random_state,
            "allow_writing_files": False,
        }
        model = CatBoostClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    return CatBoostClassifier(
        **study.best_params, verbose=0, random_state=random_state, allow_writing_files=False
    )


def _search_xgboost(X, y, cv, n_trials, random_state):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": random_state,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    return XGBClassifier(**study.best_params, eval_metric="logloss", random_state=random_state)


def _search_random_forest(X, y, cv, param_grid, random_state):
    model = RandomForestClassifier(random_state=random_state)
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_


def build_voting_classifier(X, y, cv, search_config, random_state=42, verbose=True):
    n_trials = search_config["n_trials"]

    if verbose:
        print("Searching LightGBM hyperparameters...")
    best_lgbm = _search_lightgbm(X, y, cv, n_trials, random_state)

    if verbose:
        print("Searching CatBoost hyperparameters...")
    best_cb = _search_catboost(X, y, cv, n_trials, random_state)

    if verbose:
        print("Searching XGBoost hyperparameters...")
    best_xgb = _search_xgboost(X, y, cv, n_trials, random_state)

    if verbose:
        print("Searching Random Forest hyperparameters...")
    best_rf = _search_random_forest(X, y, cv, search_config["rf_param_grid"], random_state)

    voting = VotingClassifier(
        estimators=[
            ("lgbm", best_lgbm),
            ("catboost", best_cb),
            ("xgboost", best_xgb),
            ("rf", best_rf),
        ],
        voting="soft",
    )
    return voting