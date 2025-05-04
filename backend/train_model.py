import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs("models/saved_models", exist_ok=True)

print("Loading datasets...")
# Load your datasets
df_yield = pd.read_csv("/Users/ash/CropYield_estimation/backend/DATASET/yield.csv")
df_temp = pd.read_csv("/Users/ash/CropYield_estimation/backend/DATASET/temp.csv")
df_rainfall = pd.read_csv("/Users/ash/CropYield_estimation/backend/DATASET/rainfall.csv")
df_pesticides = pd.read_csv("/Users/ash/CropYield_estimation/backend/DATASET/pesticides.csv")
df_yield_df = pd.read_csv("/Users/ash/CropYield_estimation/backend/DATASET/yield_df.csv")

# Data preprocessing
print("Data Preprocessing...")
if 'Unnamed: 0' in df_yield_df.columns:
    df_yield_df.drop(['Unnamed: 0'], axis=1, inplace=True)
df_rainfall.rename(columns={' Area':'Area'}, inplace=True)
df_temp.rename(columns={'year':'Year', 'country':'Area'}, inplace=True)

# Merge datasets
df_temprain = pd.merge(df_rainfall, df_temp, on=['Year','Area'])
df_trp = pd.merge(df_temprain, df_pesticides, on=['Year','Area'])
yield_df = pd.merge(df_yield_df, df_yield, on=['Year','Area','Item'])

# Clean and prepare the final dataset
Yield_final_data = yield_df.copy()
columns_to_drop = ['Area Code', 'Year Code', 'Domain', 'Domain Code', 'Item Code', 
                   'Element', 'Element Code', 'Unit', 'Value']
Yield_final_data.drop(columns_to_drop, axis=1, inplace=True)

# Split 'Item' column with multiple crops
Yield_final_data['Item'] = Yield_final_data['Item'].str.split(', ')
Yield_final_data = Yield_final_data.explode('Item').reset_index(drop=True)

print(f"Dataset shape after preprocessing: {Yield_final_data.shape}")

# Feature Engineering
print("Feature Engineering...")

# 1. Create binary features for each crop type
crop_dummies = pd.get_dummies(Yield_final_data['Item'], prefix='crop')
Yield_final_data = pd.concat([Yield_final_data, crop_dummies], axis=1)

# 2. Create interaction features
Yield_final_data['temp_rain_interaction'] = Yield_final_data['avg_temp'] * Yield_final_data['average_rain_fall_mm_per_year']
Yield_final_data['pesticide_per_temp'] = Yield_final_data['pesticides_tonnes'] / (Yield_final_data['avg_temp'] + 1e-6)
Yield_final_data['rain_squared'] = Yield_final_data['average_rain_fall_mm_per_year'] ** 2
Yield_final_data['temp_squared'] = Yield_final_data['avg_temp'] ** 2

# 3. Log transform skewed features
Yield_final_data['log_pesticides'] = np.log1p(Yield_final_data['pesticides_tonnes'])
Yield_final_data['log_yield'] = np.log1p(Yield_final_data['hg/ha_yield'])

# 4. Create temporal features
Yield_final_data['year_squared'] = Yield_final_data['Year'] ** 2
Yield_final_data['year_normalized'] = (Yield_final_data['Year'] - Yield_final_data['Year'].min()) / (Yield_final_data['Year'].max() - Yield_final_data['Year'].min())

# Prepare data for modeling
y = Yield_final_data['log_yield']  # Using log-transformed yield
X = Yield_final_data.drop(['hg/ha_yield', 'Item', 'Area', 'log_yield'], axis=1)

# Identify categorical and numerical columns
categorical_cols = [c for c in X.columns if c.startswith('crop_')]
numerical_cols = [c for c in X.columns if c not in categorical_cols]

print(f"Number of features: {X.shape[1]} (Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)})")

# Create preprocessing pipeline for feature selection
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Split data for feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after preprocessing
numeric_features = numerical_cols
categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numeric_features, categorical_features.tolist()])

# Train Random Forest to get feature importances
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_train_preprocessed, y_train)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_selector.feature_importances_
}).sort_values('Importance', ascending=False)

# Select top features
top_n_features = 15  # Adjust based on feature importance results
top_features = feature_importances.head(top_n_features)['Feature'].tolist()

print(f"Selected top {top_n_features} features:")
for i, feature in enumerate(top_features):
    print(f"{i+1}. {feature}")

# Extract indices of top features from all features
top_indices = [np.where(all_features == feature)[0][0] for feature in top_features]

# Create filtered datasets with only top features
X_train_selected = X_train_preprocessed[:, top_indices]
X_test_selected = X_test_preprocessed[:, top_indices]

# Train Random Forest model (best performer from notebook)
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_selected)
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

# Save trained model, preprocessor, and feature names
print("Saving model files...")
joblib.dump(rf_model, "models/saved_models/random_forest_model.pkl")
joblib.dump(preprocessor, "models/saved_models/preprocessor.pkl")
joblib.dump(top_features, "models/saved_models/top_features.pkl")
joblib.dump(X.columns.tolist(), "models/saved_models/feature_names.pkl")

print("Model and associated files have been saved to models/saved_models/")
