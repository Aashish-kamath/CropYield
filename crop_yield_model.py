import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(yield_path, temp_path, rainfall_path, pesticides_path):
    """
    Load and preprocess the datasets for model training
    """
    logger.info("Loading datasets...")
    
    # Load datasets
    df_yield = pd.read_csv(yield_path)
    df_temp = pd.read_csv(temp_path)
    df_rainfall = pd.read_csv(rainfall_path)
    df_pesticides = pd.read_csv(pesticides_path)
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    
    # Clean column names
    df_rainfall.rename(columns={' Area':'Area'}, inplace=True)
    df_temp.rename(columns={'year':'Year', 'country':'Area'}, inplace=True)
    
    # Merge datasets
    df_temprain = pd.merge(df_rainfall, df_temp, on=['Year','Area'])
    df_trp = pd.merge(df_temprain, df_pesticides, on=['Year','Area'])
    yield_df = pd.merge(df_trp, df_yield, on=['Year','Area'])
    
    # Clean and prepare the final dataset
    Yield_final_data = yield_df.copy()
    
    # Drop unnecessary columns
    columns_to_drop = ['Area Code', 'Year Code', 'Domain', 'Domain Code', 'Item Code', 
                       'Element', 'Element Code', 'Unit', 'Value']
    
    for col in columns_to_drop:
        if col in Yield_final_data.columns:
            Yield_final_data.drop(col, axis=1, inplace=True)
    
    # Split 'Item' column with multiple crops if necessary
    if Yield_final_data['Item'].str.contains(',').any():
        Yield_final_data['Item'] = Yield_final_data['Item'].str.split(', ')
        Yield_final_data = Yield_final_data.explode('Item').reset_index(drop=True)
    
    logger.info(f"Final dataset shape: {Yield_final_data.shape}")
    
    return Yield_final_data

def engineer_features(df):
    """
    Perform feature engineering on the dataset
    """
    logger.info("Performing feature engineering...")
    
    # Create binary features for each crop type
    crop_dummies = pd.get_dummies(df['Item'], prefix='crop')
    df = pd.concat([df, crop_dummies], axis=1)
    
    # Create interaction features
    df['temp_rain_interaction'] = df['avg_temp'] * df['average_rain_fall_mm_per_year']
    df['pesticide_per_temp'] = df['pesticides_tonnes'] / (df['avg_temp'] + 1e-6)
    df['rain_squared'] = df['average_rain_fall_mm_per_year'] ** 2
    df['temp_squared'] = df['avg_temp'] ** 2
    
    # Log transform skewed features
    df['log_pesticides'] = np.log1p(df['pesticides_tonnes'])
    df['log_yield'] = np.log1p(df['hg/ha_yield'])
    
    # Create temporal features
    df['year_squared'] = df['Year'] ** 2
    df['year_normalized'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
    
    return df

def prepare_training_data(df):
    """
    Prepare data for model training
    """
    logger.info("Preparing training data...")
    
    # Define target and features
    y = df['log_yield']  # Using log-transformed yield
    X = df.drop(['hg/ha_yield', 'Item', 'Area', 'log_yield'], axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = [c for c in X.columns if c.startswith('crop_')]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    
    logger.info(f"Features: {len(X.columns)} (Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])
    
    return X_train, X_test, y_train, y_test, preprocessor, X.columns.tolist()

def train_model(X_train, y_train, preprocessor):
    """
    Train the XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    # Preprocess the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # Initialize and train XGBoost model
    model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train_preprocessed, y_train)
    
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test, preprocessor):
    """
    Evaluate the trained model
    """
    logger.info("Evaluating model...")
    
    # Preprocess the test data
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred_log = model.predict(X_test_preprocessed)
    
    # Convert back from log scale
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred_log)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_exp, y_pred_exp)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_exp, y_pred_exp)
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    
    logger.info(f"RMSE: {rmse:.3f}")
    logger.info(f"R2: {r2:.3f}")
    logger.info(f"MAE: {mae:.3f}")
    
    return rmse, r2, mae, y_test_exp, y_pred_exp

def save_model(model, preprocessor, feature_names, output_dir="saved_models"):
    """
    Save the trained model and preprocessor
    """
    logger.info(f"Saving model to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model and preprocessor
    joblib.dump(model, os.path.join(output_dir, "xgboost_model.pkl"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))
    
    logger.info("Model saved successfully")

def load_model(model_path, preprocessor_path, feature_names_path):
    """
    Load the trained model and preprocessor
    """
    logger.info(f"Loading model from {model_path}...")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = joblib.load(feature_names_path)
    
    logger.info("Model loaded successfully")
    
    return model, preprocessor, feature_names

def run_training_pipeline(yield_path, temp_path, rainfall_path, pesticides_path, output_dir="saved_models"):
    """
    Run the complete training pipeline
    """
    # Load and preprocess data
    data = load_and_preprocess_data(yield_path, temp_path, rainfall_path, pesticides_path)
    
    # Engineer features
    data_engineered = engineer_features(data)
    
    # Prepare training data
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_training_data(data_engineered)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    rmse, r2, mae, y_test_actual, y_test_pred = evaluate_model(model, X_test, y_test, preprocessor)
    
    # Save model
    save_model(model, preprocessor, feature_names, output_dir)
    
    return model, preprocessor, feature_names, rmse, r2

if __name__ == "__main__":
    # Example usage
    yield_path = "/Users/ash/CropYield_estimation /DATASET/yield.csv"
    temp_path = "/Users/ash/CropYield_estimation /DATASET/temp.csv"
    rainfall_path = "/Users/ash/CropYield_estimation /DATASET/rainfall.csv"
    pesticides_path = "/Users/ash/CropYield_estimation /DATASET/pesticides.csv"
    
    run_training_pipeline(yield_path, temp_path, rainfall_path, pesticides_path)