# same as EDA.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# ================== lodad data ==================
train = pd.read_csv("data/raw/train.csv")

# ================== missing value ==================
none_fill_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
    "MasVnrType"
]

for col in none_fill_cols:
    train[col] = train[col].fillna("None")

for col in train.select_dtypes(include="object").columns:
    if train[col].isna().sum() > 0 and col not in none_fill_cols:
        train[col] = train[col].fillna(train[col].mode()[0])

# ================== outlier ==================
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

train = remove_outliers_iqr(train, "SalePrice")

# ================== VIF ==================
def reduce_vif(df, threshold=5.0, verbose=True):
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=["SalePrice"], errors="ignore")
    df_numeric = df_numeric.dropna(axis=1)
    removed_features = []
    iteration = 1

    while True:
        X = add_constant(df_numeric)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        vif = vif.drop("const")
        max_vif = vif.max()
        max_feature = vif.idxmax()

        if verbose:
            print(f"Iteration {iteration}")
            print(vif.sort_values(ascending=False).head(10))
            print("-" * 40)

        if max_vif <= threshold:
            break

        df_numeric.drop(columns=[max_feature], inplace=True)
        removed_features.append(max_feature)
        iteration += 1

    return df_numeric.columns.tolist(), removed_features

numeric_cols_to_keep, _ = reduce_vif(train)

# ================== Numric normalization ==================
scaler = MinMaxScaler()
train[numeric_cols_to_keep] = scaler.fit_transform(train[numeric_cols_to_keep])

# ================== categorical Encoding ==================
train_encoded = pd.get_dummies(train, columns=train.select_dtypes(include="object").columns)

# ================== output ==================
os.makedirs("../data/processed", exist_ok=True)
train_encoded.to_csv("../data/processed/train_encoded.csv", index=False)
print("save to：data/processed/train_encoded.csv")
