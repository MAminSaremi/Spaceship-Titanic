# 🚀 Spaceship Titanic — Transported Prediction

Predicting whether a passenger was transported to another dimension during the Spaceship Titanic incident, using advanced feature engineering, EDA-driven imputation, and a stacked ensemble of multiple machine learning models.

## 📋 Project Features

### Feature Engineering
- Extracted `deck`, `num`, `side` from the `Cabin` column
- Extracted `groupId` from `PassengerId` to identify passengers traveling together
- Computed `spending` (sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
- Extracted surname (`Surname`) and identified family members by combining `groupId` + `Surname`
- Built derived features: `log_spending` (log transform due to severe skewness), `is_child` (Age ≤ 12), `group_size` (number of members per group)

### EDA-Driven Smart Imputation
Instead of simply filling missing values with the global mean/mode, this project uses patterns discovered through exploratory data analysis, including:
- `HomePlanet`: based on the group (`groupId`) mode, then based on the joint condition of `VIP` and `CryoSleep`
- `CryoSleep`: based on its relationship with `spending` (no passenger with positive spending was in CryoSleep)
- `deck` and `side`: based on the mode within each group/family, then based on the combination of `HomePlanet` + `CryoSleep`
- `Age`: imputed using several conditional rules (based on family, spending pattern, and the group median of `HomePlanet`/`CryoSleep`)
- `VIP`, `Destination`: based on the mode within each family, then based on `HomePlanet`

> ⚠️ Important note: throughout the imputation process, the target variable (`Transported`) was never directly used to fill missing values, to avoid **target leakage** — even in cases where a strong statistical pattern existed.

### Modeling
- Categorical features (`HomePlanet`, `Destination`, `deck`, `side`) preprocessed with `OneHotEncoder` inside a `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV` for four base models:
  - CatBoost
  - XGBoost
  - Random Forest
  - LightGBM
- Models combined using `StackingClassifier` (with a Random Forest meta-model)
- Final evaluation via Cross Validation (5-Fold Stratified) for a more stable performance estimate

## 📊 Dataset

Required file: `train.csv` (the standard dataset from the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic))

Main columns:

| Column | Description |
| --- | --- |
| `PassengerId` | Passenger ID (includes group and number) |
| `HomePlanet` | Passenger's planet of origin |
| `CryoSleep` | Whether the passenger was in cryosleep |
| `Cabin` | Cabin number (deck/num/side) |
| `Destination` | Travel destination |
| `Age` | Passenger's age |
| `VIP` | VIP membership |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Passenger spending across various ship amenities |
| `Name` | Passenger's full name |
| `Transported` | Target variable — whether the passenger was transported |

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn catboost xgboost lightgbm
```

## 🚀 How to Run

1. Place the `train.csv` file in the same directory as the script.
2. Run the script:

```bash
python main.py
```

3. The stacking model is trained on the training data and then evaluated using cross validation.
4. The output includes the accuracy of each fold, the mean accuracy, the standard deviation, and the 95% confidence interval.

## 📈 Model Results (GridSearchCV)

| Model | Train Accuracy | Test Accuracy | Gap |
| --- | --- | --- | --- |
| CatBoost | 77.54% | 76.02% | 0.0152 |
| XGBoost | 78.04% | 76.02% | 0.0202 |
| Random Forest | 82.84% | 76.14% | 0.0671 |
| LightGBM | 82.59% | 76.31% | 0.0628 |

### Final Model: Stacking (CatBoost + XGBoost + LightGBM → Random Forest)

| Metric | Value |
| --- | --- |
| Train Accuracy | 79.12% |
| Test Accuracy | 76.54% |
| Gap | 0.0258 |

> 📌 Among the meta-models tested, **Random Forest** performed best in the final ensemble.
> 📌 Experiments showed no meaningful difference between CatBoost with and without One-Hot Encoding; however, to maintain consistency within the stacking framework, a pipeline including `OneHotEncoder` was used for all base models.

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn (Pipeline, ColumnTransformer, StackingClassifier, GridSearchCV)
- CatBoost
- XGBoost
- LightGBM

## 📁 Project Structure

```
.
├── main.py        # Main project script (preprocessing, feature engineering, modeling)
├── train.csv       # Training dataset (must be provided separately)
└── README.md
```

## 📝 Additional Notes

- The `GridSearchCV` blocks for all four base models are kept commented out in the code so they can be re-run if needed (they are time-consuming); the best parameters found were applied directly in each pipeline's definition.
- Categorical preprocessing is handled through a `ColumnTransformer` inside each pipeline rather than manually on the dataframe — this avoids mutating the raw data directly and ensures the same transformations can be consistently applied to new data (e.g. `test.csv`).
- Final model evaluation was performed with `cross_val_score` using 5-Fold Stratified cross validation, for a more stable accuracy estimate that's less dependent on a single train/test split.

## 📄 License

This project was created for educational/personal purposes.
