
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy import stats

titanic_data = sns.load_dataset('titanic')
print("Dataset shape:", titanic_data.shape)
print(titanic_data.info())
print(titanic_data.head())
print("Number of duplicates in the dataset:", titanic_data.duplicated().sum())

titanic_data = titanic_data.drop_duplicates()
print("Dataset shape after removing duplicates:", titanic_data.duplicated().sum())

X = titanic_data.drop(columns=['survived'])
y = titanic_data['survived']

nums = X.select_dtypes(include=['number']).columns.tolist()
cats = X.select_dtypes(include=['object', 'category']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nums_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cats_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('nums', nums_pipeline, nums),
    ('cats', cats_pipeline, cats)
])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print("Training data shape:", X_train_preprocessed.shape)
print("Testing data shape:", X_test_preprocessed.shape)
print("Preprocessing completed successfully.")
