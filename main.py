import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
df = pd.read_csv("train.csv")

# print(df.head())
# print(df.isnull().sum())
# !---------------------------------------
df['deck'] = df.Cabin.str.extract(r'^(.)')
df['num'] = df.Cabin.str.extract(r'(\d+)')
df['side'] = df.Cabin.str.split('/').str[2]
df['groupId'] = df.PassengerId.str.extract(r'(\d\d\d\d)')
df['Passenger_Id'] = df.PassengerId.str.extract(r'(\d{2}$)')
df = df.drop('Cabin' , axis=1)
df = df.drop('PassengerId' , axis=1 )

# !---------------------------------------
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['spending'] = df[cols].fillna(0).sum(axis=1)

df = df.drop(['RoomService' , 'FoodCourt' , 'ShoppingMall' , 'Spa' , 'VRDeck']  , axis=1)
df['Surname'] = df['Name'].str.split().str[-1]
df['family'] = df.groupby(['groupId', 'Surname']).ngroup()


# !---------------------------------------
col = ['spending']
mask = (
    df['CryoSleep'] == True)
df.loc[mask, col] = df.loc[mask, col].fillna(0)
# !---------------------------------------
# !---------------------------------------
# print(df['HomePlanet'].unique())
# print(df.loc[df['HomePlanet'].isna()])
# print(df.groupby('groupId')['HomePlanet'].nunique().value_counts())

#? Data exploration shows a strong relationship between HomePlanet and groupId.
#? Members of the same group almost always originate from the same HomePlanet.
def fill_mode(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else None

df['HomePlanet'] = df['HomePlanet'].fillna(
    df.groupby('groupId')['HomePlanet'].transform(fill_mode)
)

# print(df.loc[(df['VIP']==True) & (df['CryoSleep']==True), 'HomePlanet'])
# mask = (df['VIP'] == True) & (df['CryoSleep'] == True)
# print(df.loc[mask, 'HomePlanet'].value_counts(dropna=False))

#? Based on EDA, all observed passengers with CryoSleep == True
#? and VIP == True have HomePlanet == 'Europa' (20/21 cases).
#? This pattern is used to impute some missing HomePlanet values
#? when these conditions are satisfied.
maskn = (
    (df['VIP'] == True) &
    (df['CryoSleep'] == True)
)

df.loc[maskn, 'HomePlanet'] = df.loc[maskn, 'HomePlanet'].fillna('Europa')




# print(df[df['HomePlanet'].isna()])
# df['AgeGroup'] = pd.cut(
#     df['Age'],
#     bins=[0,12,18,30,50,80]
# )
# print(
#     pd.crosstab(
#         df['AgeGroup'],
#         df['HomePlanet'],
#         normalize='index'
#     ) * 100
# )
#? After applying the previous imputation rules, only 110 missing
#? HomePlanet values remained. No meaningful pattern was identified
#? among these records. Age-group analysis showed no strong relationship
#? between Age and HomePlanet, except for passengers aged 0–12.
#? Since all remaining missing records belonged to passengers older
#? than 18, the remaining HomePlanet values were imputed using the
#? overall mode of the dataset.
df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
df['Destination'] = (
    df.groupby('HomePlanet')['Destination']
      .transform(lambda x: x.fillna(x.mode().iloc[0]))
)
# !---------------------------------------

# !---------------------------------------
# print(df.loc[df['spending'] > 0, 'CryoSleep'].value_counts(dropna=False))
# print(df[(df['spending'] > 0) & (df['CryoSleep'] == True)])
#? Based on EDA, no passenger with spending > 0
#? was found to be in CryoSleep. Therefore, missing
#? CryoSleep values are filled with False when
#? spending is greater than zero.

maskm = (
    (df['spending'] > 0 )
)
df.loc[maskm , 'CryoSleep'] = df.loc[maskm , 'CryoSleep'].fillna(False)
# !
# print(df.loc[df['CryoSleep'].isna(), 'Transported'].value_counts(dropna=False))
#? EDA showed that approximately 82% of passengers with
#? CryoSleep == True were transported.
#? Although this relationship is strong, it is intentionally
#? not used for imputing CryoSleep because Transported is
#? the target variable and using it would introduce target leakage.
# msk = (
#     (df['CryoSleep'].isna()) &
#     (df['Transported'] == True)
# )
# df.loc[msk , 'CryoSleep'] = df.loc[msk , 'CryoSleep'].fillna(True)
# !
# tmp = df[df['spending'] == 0]
# print(tmp['CryoSleep'].value_counts(normalize=True) * 100)
#?Based on EDA, about 85% of passengers with spending == 0
#?were in CryoSleep. This pattern is used to impute missing
#?CryoSleep values when spending is zero.
msk = (
    (df['CryoSleep'].isna()) &
    (df['spending'] == 0)
)
df.loc[msk , 'CryoSleep'] = df.loc[msk , 'CryoSleep'].fillna(True)
# !---------------------------------------

# !---------------------------------------
# print(df.groupby('groupId')['side'].nunique(dropna=True).value_counts())
# print(df.groupby('groupId')['num'].nunique(dropna=True).value_counts())
#?EDA showed that passengers within the same group
#?almost always share the same cabin side.
#?Therefore, missing side values are filled using
#?the group-wise mode.
df['side'] = df.groupby('groupId')['side'].transform(
    lambda x : x.fillna(x.mode()[0]) if not x.mode().empty else x
)

df['num'] = df.groupby('groupId')['num'].transform(
    lambda x : x.fillna(x.mode()[0]) if not x.mode().empty else x
)
df['num'] = df['num'].fillna(df['num'].mode()[0])

# tmp = df[
#     (df['HomePlanet'] == 'Earth') &
#     (df['CryoSleep'] == True)
# ]
# print(tmp['deck'].value_counts(normalize=True) * 100)
#? EDA showed that approximately 97% of passengers with
#? HomePlanet == 'Earth' and CryoSleep == True
#? were assigned to deck 'G'. Therefore, missing
#? deck values are imputed as 'G' under these conditions.

# tmp = df[
#     (df['HomePlanet'] == 'Mars') &
#     (df['CryoSleep'] == True) &
#     (df['Age'] <= 20)
# ]
# print(tmp['deck'].value_counts(normalize=True) * 100)
#? EDA showed that approximately 97% of passengers with
#? HomePlanet == 'Earth' and CryoSleep == True
#? were assigned to deck 'G'. Therefore, missing
#? deck values are imputed as 'G' under these conditions.

msk_deck = (
    (df['HomePlanet'] == 'Earth') &
    (df['CryoSleep'] == True)
)
df.loc[msk_deck , 'deck'] = df.loc[msk_deck , 'deck'].fillna('G')

msk_deck2 = (
    (df['HomePlanet'] == 'Mars') &
    (df['CryoSleep'] == True)&
    (df['Age'] <= 20)
)
df.loc[msk_deck2 , 'deck'] = df.loc[msk_deck2 , 'deck'].fillna('F')

# !---------------------------------------
# !---------------------------------------
# result = df[df['Age'].between(1, 20)].groupby('Age')['spending'].mean()
# print(result)
#? Based on EDA, no passenger aged 12 or younger
#? had any spending. Therefore, spending is imputed
#? as 0 for passengers in this age group.
msk_spending = (
    (df['Age'] <= 12)
)

df.loc[msk_spending, 'spending'] = 0
# !---------------------------------------
# !---------------------------------------
# tmp = df[(df['Age'] <= 20) &(df['VIP'].notna())]
# print(tmp['VIP'].value_counts())
# print(tmp['VIP'].value_counts(normalize=True) * 100)
#? EDA revealed that almost no passengers under the age
#? of 20 were VIP. Therefore, missing VIP values are
#? imputed as False for passengers in this age group.
msk_vip = (

    (df['Age'] <= 20)
)
df.loc[msk_vip, 'VIP'] = df.loc[msk_vip, 'VIP'].fillna(False)
# !---------------------------------------

# !---------------------------------------
# tmp = df[(df['HomePlanet'] == 'Mars') &(df['VIP'] == False) &(df['Destination'].notna())]
# print(tmp['Destination'].value_counts())
# print(tmp['Destination'].value_counts(normalize=True) * 100)
#? EDA showed that in more than 85% of cases,
#? passengers with HomePlanet == 'Mars' and VIP == False
#? had Destination == 'TRAPPIST-1e'. Therefore, missing
#? Destination values are imputed as 'TRAPPIST-1e'
#? under these conditions.
msk_des = (
    (df['HomePlanet'] == 'Mars') &
    (df['VIP'] == False)
)
df.loc[msk_des, 'Destination'] = df.loc[msk_des, 'Destination'].fillna('TRAPPIST-1e')
# !---------------------------------------

# !---------------------------------------

family_size = df.groupby('family')['family'].transform('size')
# tmp = df[(family_size > 1) &(df['Age'].notna()) &(df['VIP'] == False) &(df['CryoSleep'] == False) &(df['spending'] == 0)]
# print(f"Count: {len(tmp)}")
# print(f"Child (Age <= 12) rate: {(tmp['Age'] <= 12).mean() * 100:.2f}%")
# print(tmp['Age'].describe())
# print(tmp['Age'].value_counts().sort_index())
#? Based on EDA:
#? In multi-member families, passengers with VIP=False, CryoSleep=False,
#? and spending=0 are children (Age <= 12) in about 90% of cases.
#? Missing ages for this pattern are therefore imputed with 6.
msk_age = (
    (family_size>1)&
    df['Age'].isna()&
    (df['VIP']==False)&
    (df['CryoSleep'] ==False)&
    (df['spending'] ==0)
)
df.loc[msk_age , 'Age']= 6
# !---------------------------------------

# !---------------------------------------
# tmp = (df[df['VIP'].notna()].groupby('family')['VIP'].nunique())
# print(tmp.value_counts().sort_index())
# print(tmp.value_counts(normalize=True).sort_index() * 100)
#?Based on EDA, family members share the same values for features such as
#?VIP, Deck,Destination and also age in more than 85% of cases.
#?Therefore, missing values are imputed using the mode within each family,
#?which provides a more accurate estimate than using the global mode
#?of the entire dataset.
def fill_vip(group):
    known = group.dropna()
    if len(known)== 0 :
        return group
    return group.fillna(known.mode()[0])
mask_ = df['family'].notna()
df.loc[mask_,'VIP'] = (
    df.loc[mask_].groupby('family')['VIP'].transform(fill_vip)
)

# tmp = (df[df['deck'].notna()].groupby('family')['deck'].nunique())
# print(tmp.value_counts().sort_index())
# print(tmp.value_counts(normalize=True).sort_index() * 100)
def fill_deck(group):
    known = group.dropna()
    if len(known)== 0 :
        return group
    return group.fillna(known.mode()[0])
mask_ = df['family'].notna()
df.loc[mask_,'deck'] = (
    df.loc[mask_].groupby('family')['deck'].transform(fill_deck)
)

# tmp = (df[df['Age'].notna()].groupby('family')['Age'].nunique())
# print(tmp.value_counts().sort_index())
# print(tmp.value_counts(normalize=True).sort_index() * 100)
def fill_Age(group):
    known = group.dropna()
    if len(known)== 0 :
        return group
    return group.fillna(known.median())
mask_ = df['family'].notna()
df.loc[mask_,'Age'] = (
    df.loc[mask_].groupby('family')['Age'].transform(fill_Age)
)

# tmp = (df[df['Destination'].notna()].groupby('family')['Destination'].nunique())
# print(tmp.value_counts().sort_index())
# print(tmp.value_counts(normalize=True).sort_index() * 100)
def fill_Destination(group):
    known = group.dropna()
    if len(known)== 0 :
        return group
    return group.fillna(known.mode()[0])
mask_ = df['family'].notna()
df.loc[mask_,'Destination'] = (
    df.loc[mask_].groupby('family')['Destination'].transform(fill_Destination)
)

# !---------------------------------------
# print(df['VIP'].value_counts())
# print(df['VIP'].value_counts(normalize=True) * 100)
#? Based on EDA, 97.69% of passengers are not VIP.
#? No meaningful common pattern was found among the remaining VIP missing values,
#? so they could not be imputed using a more specific rule.
#? Therefore, the remaining missing values are filled with False.
df['VIP'] = df['VIP'].fillna(False)
# !---------------------------------------
# tmp = df[(df['HomePlanet'] == 'Earth') &(df['VIP'] == False) &(df['CryoSleep'] == False) &(df['spending'] == 0) &(df['Age'] <= 12)]
# print(tmp['Age'].describe())
# print(tmp['Age'].value_counts().sort_index())
#? Based on EDA, all 291 passengers matching this pattern
#? (Earth, VIP=False, CryoSleep=False, spending=0)
#? are between 0 and 12 years old.
#? Since the median age is 6, missing ages for this pattern
#? are imputed with 6.
msk__age = (
    df['Age'].isna() &
    (df['HomePlanet'] == 'Earth') &
    (df['VIP'] == False) &
    (df['CryoSleep'] == False) &
    (df['spending'] == 0)
)

df.loc[msk__age, 'Age'] = 6

# ? Based on EDA, Age distributions differ across (HomePlanet, CryoSleep) groups
#? (e.g., median age ranges from 21 to 34).
#? Therefore, the remaining missing ages are imputed using the median age
#? of each (HomePlanet, CryoSleep) group instead of the global median.
df['Age'] = df['Age'].fillna(
    df.groupby(['HomePlanet','CryoSleep'])['Age']
      .transform('median')
)
# !---------------------------------------
# !---------------------------------------
tmp = df[df['side'].notna()]

df['side'] = df['side'].fillna(df['side'].mode()[0])

# !---------------------------------------
#? Based on EDA, HomePlanet and CryoSleep provide the best available
#? grouping for estimating missing Deck values. Although the strength
#? of the pattern varies across groups, using the mode within each
#? (HomePlanet, CryoSleep) group provides a more informative estimate
#? than using the global mode of the entire dataset.
def fill_deck(group):
    m = group.mode()
    return group.fillna(m.iloc[0]) if not m.empty else group

df['deck'] = (
    df.groupby(['HomePlanet','CryoSleep'])['deck']
      .transform(fill_deck)
)
# !---------------------------------------
# print(pd.crosstab(df['HomePlanet'],df['Destination'],normalize='index') * 100)
#? Based on EDA, each HomePlanet has its own dominant Destination.
#? Therefore, missing Destination values are filled with the mode
#? of the corresponding HomePlanet, providing a more accurate
#? estimate than using the global mode of the dataset.
df['Destination'] = (
    df.groupby('HomePlanet')['Destination']
      .transform(lambda x: x.fillna(x.mode()[0]))
)
# !-----------------------------------------------------------------------------------------------------------------------------------------
df['Surname'] = df['Surname'].fillna('Unknown')
df['family'] = df['family'].fillna(-1)

# skewness = train_x.select_dtypes(include='number').skew().sort_values(ascending=False)
# print(skewness)
#? Since the spending feature exhibited severe right skewness,
#? a log1p transformation was applied to reduce skewness
#? and improve model performance.
df['log_spending'] = np.log1p(df['spending'])
df['is_child']= (df['Age'] <=12).astype(int)
df["group_size"] = df.groupby("groupId")["groupId"].transform("size")

x = df[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'is_child','VIP',
       'deck', 'num', 'side', 'groupId','log_spending','family']]

y = df['Transported']

chat_feature = ['HomePlanet' , 'Destination' , 'deck' , 'side']
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            chat_feature
        )
    ],
    remainder="passthrough"
)

train_x , test_x , train_y , test_y = train_test_split(x , y ,test_size=0.2 , random_state=42 , stratify=y)


# ! ------------------------------------------------CATBOOST-----------------------------------------------
#? The following results were obtained using GridSearchCV.
#? Best Score : 0.7598510465531241
#? Best Params : {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'subsample': 0.8}
#? Train Accuracy : 0.7754
#? Test Accuracy  : 0.7602
#? Gap            : 0.0152
#* Experiments showed no significant performance difference
#* between CatBoost with One-Hot Encoding and CatBoost alone.
#* Therefore, a pipeline including OneHotEncoder was adopted
#* to maintain compatibility within the stacking framework.
pipe_cat = Pipeline([
    ("preprocess", preprocessor),
    # The following hyperparameters were configured according to the best configuration found by GridSearchCV.
    ("model",CatBoostClassifier(depth= 4, iterations= 500, l2_leaf_reg =3, learning_rate =0.03, subsample =0.8))])
# ! ------------------------------------------------CATBOOST-----------------------------------------------
# ! ------------------------------------------------XGBoost-----------------------------------------------



from sklearn.metrics import accuracy_score

def mp_finder(model, parameters):

    grid = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
        verbose=2
        
    )

    grid.fit(train_x, train_y)

    train_pred = grid.predict(train_x)
    test_pred = grid.predict(test_x)

    train_acc = accuracy_score(train_y, train_pred)
    test_acc = accuracy_score(test_y, test_pred)

    print(f"Best Params: {grid.best_params_}")
    print("-" * 40)
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Gap            : {train_acc - test_acc:.4f}")

    return train_acc, test_acc

# xg_param_grid = {
#     "n_estimators": [300, 500, 700],
#     "max_depth": [4, 5, 6],
#     "learning_rate": [0.03, 0.05, 0.1],
#     "subsample": [0.8, 1.0],
#     "colsample_bytree": [0.8, 1.0],
#     "min_child_weight": [1, 3],
#     "gamma": [0, 0.1],
#     "reg_alpha": [0, 0.1],
#     "reg_lambda": [1, 3]
# }
pipe_xg = Pipeline([
    ("preprocess", preprocessor),
    # The following hyperparameters were configured according to the best configuration found by GridSearchCV.
    ("model",XGBClassifier(colsample_bytree= 0.8, gamma= 0, learning_rate= 0.03, max_depth= 4, min_child_weight= 3, n_estimators= 300, reg_alpha= 0, reg_lambda= 3, subsample =0.8))])
# xg_train_acc , xg_test_acc = mp_finder(pipe_xg , xg_param_grid)
#? The following results were obtained using GridSearchCV.
#? Best Params: {'model__colsample_bytree': 0.8, 'model__gamma': 0, 'model__learning_rate': 0.03, 'model__max_depth': 4, 'model__min_child_weight': 3, 'model__n_estimators': 300, 'model__reg_alpha': 0, 'model__reg_lambda': 3, 'model__subsample': 0.8}
#? ----------------------------------------
#? Train Accuracy : 0.7804
#? Test Accuracy  : 0.7602
#? Gap            : 0.0202
# ! ------------------------------------------------XGBoost-----------------------------------------------

# ! ------------------------------------------------RandomForest-----------------------------------------------
# rf_param_grid = {
#     "model__n_estimators": [200, 500, 800],
#     "model__max_depth": [None, 10, 20, 30],
#     "model__min_samples_split": [2, 5, 10],
#     "model__min_samples_leaf": [1, 2, 4],
#     "model__max_features": ["sqrt", "log2"],
#     "model__bootstrap": [True, False]
# }
pipe_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model",RandomForestClassifier())])
# rf_train_acc ,rf_test_acc = mp_finder(pipe_rf ,rf_param_grid)
#? The following results were obtained using GridSearchCV.
#? Best Params: {'model__bootstrap': False, 'model__max_depth': 10, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 1, 'model__min_samples_split': 10, 'model__n_estimators': 200}
#? ----------------------------------------
#? Train Accuracy : 0.8284
#? Test Accuracy  : 0.7614
#? Gap            : 0.0671
# ! ------------------------------------------------RandomForest-----------------------------------------------
# ! ------------------------------------------------LightGBM-----------------------------------------------
# lgbm_param_grid = {
#     "model__n_estimators": [300, 500],
#     "model__learning_rate": [0.03, 0.05],
#     "model__max_depth": [-1, 6],
#     "model__num_leaves": [31, 63],
#     "model__min_child_samples": [20, 40],
#     "model__subsample": [0.8, 1.0],
#     "model__colsample_bytree": [0.8, 1.0]
# }
pipe_lg = Pipeline([
    ("preprocess", preprocessor),
    ("model",LGBMClassifier(colsample_bytree= 0.8, learning_rate= 0.03, max_depth= 6, min_child_samples =40, n_estimators= 500, num_leaves =31, subsample =0.8))])
# lg_train_acc ,lg_test_acc = mp_finder(pipe_lg ,lgbm_param_grid)
#? Best Params: {'model__colsample_bytree': 0.8, 'model__learning_rate': 0.03, 'model__max_depth': 6, 'model__min_child_samples': 40, 'model__n_estimators': 500, 'model__num_leaves': 31, 'model__subsample': 0.8}
#? ----------------------------------------
#? Train Accuracy : 0.8259
#? Test Accuracy  : 0.7631
#? Gap            : 0.0628
# ! ------------------------------------------------LightGBM-----------------------------------------------
# ! ------------------------------------------------stacking-----------------------------------------------

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
estimators = [

    ("cat", pipe_cat),

    ("xgb", pipe_xg),

    ("lgb", pipe_lg),


]
meta_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    random_state=42
)
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
stack = StackingClassifier(

    estimators=estimators,

    final_estimator=meta_model,

    cv=cv,

    stack_method="predict_proba",

    passthrough=False,

    n_jobs=-1

)
stack.fit(train_x, train_y)
train_pred = stack.predict(train_x)

test_pred = stack.predict(test_x)
from sklearn.metrics import accuracy_score

#? The following experiment compares different models
#? to determine the best meta-model for stacking.
# print("Train :", accuracy_score(train_y, train_pred))
# print("Test  :", accuracy_score(test_y, test_pred))
# print("Gap   :", accuracy_score(train_y, train_pred) - accuracy_score(test_y, test_pred))

# ? the best model for meta_model is
# ? random forest
# Train : 0.7911993097497843
# Test  : 0.765382403680276
# Gap   : 0.02581690606950826
# test = pd.read_csv("test.csv")
# ! ------------------------------------------------stacking-----------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    estimator=stack,
    X=x,
    y=y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("=" * 40)
print("Stacking Cross Validation")
print("=" * 40)
print(f"Fold Scores : {scores}")
print(f"Mean Accuracy : {scores.mean():.4f}")
print(f"Std Accuracy  : {scores.std():.4f}")
print(f"95% CI : {scores.mean():.4f} ± {1.96 * scores.std():.4f}")