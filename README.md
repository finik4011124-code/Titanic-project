# Titanic PROJECT


## Usage

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```


## Structure

```
titanic_project/
├── data/                     # train.csv, test.csv, gender_submission.csv
├── results/
│   ├── comparison_table.csv
│   └── submission.csv
├── src/
│   ├── data.py               # loading, imputation, encoding
│   ├── features.py           # Title, FamilySize, IsAlone, Deck
│   ├── models.py             # hyperparameter search + voting ensemble
│   └── train.py              # cross-validation
├── eda.ipynb                 
├── experiments.ipynb         # preprocessing, modeling
├── config.yaml               # paths, cv settings, search config
├── main.py
└── requirements.txt
```


## Results

| Experiment                      | Accuracy | Precision | Recall | F1     | Notes                              |
|---------------------------------|----------|-----------|--------|--------|------------------------------------|
| Baseline LogReg (4 features)    | 0.715    | 0.700     | 0.540  | 0.610  | Pclass, SibSp, Parch, Fare         |
| LogReg + preprocessing          | 0.820    | 0.800     | 0.760  | 0.780  | added Sex, Age, HasCabin, Embarked |
| LogReg + GridSearchCV           | 0.798    | 0.748     | 0.713  | 0.730  | l2, C=0.1                          |
| KNN + GridSearchCV              | 0.826    | 0.794     | 0.740  | 0.765  | k=7, euclidean, uniform            |
| Decision Tree + GridSearchCV    | 0.823    | 0.791     | 0.731  | 0.759  | max_depth=10                       |
| Random Forest + GridSearchCV    | 0.838    | 0.822     | 0.740  | 0.778  | n_estimators=300, max_depth=10     |
| CatBoost + Optuna               | 0.839    | 0.833     | 0.728  | 0.777  | depth=3, lr=0.072, iters=390       |
| LightGBM + Optuna               | 0.841    | 0.833     | 0.731  | 0.778  | max_depth=6, lr=0.020, n=226       |
| XGBoost + Optuna                | 0.839    | 0.835     | 0.725  | 0.776  | max_depth=8, lr=0.012, n=315       |
| LightGBM + FE + Optuna          | 0.841    | 0.828     | 0.740  | 0.781  | added Title, FamilySize, Deck      |
| CatBoost + FE + Optuna          | 0.839    | 0.826     | 0.737  | 0.779  | depth=4, lr=0.127, iters=344       |
| XGBoost + FE + Optuna           | 0.837    | 0.828     | 0.728  | 0.774  | max_depth=5, lr=0.015, n=392       |
| DNN (PyTorch, MLP + BatchNorm)  | 0.832    | 0.833     | 0.702  | 0.761  | Adam lr=0.01, 100 epochs           |
| DNN (skorch + CosineAnnealingLR)| 0.827    | 0.815     | 0.713  | 0.760  | Adam lr=0.01, 300 epochs, BatchNorm|
| Voting (LGB+CAT+XGB+RF)         | 0.842    | 0.823     | 0.748  |0.783   | soft voting, FE features           |
