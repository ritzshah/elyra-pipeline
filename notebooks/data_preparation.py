#!/usr/bin/env python3
"""
Data Preparation Pipeline Component (Python Script Version)
This script performs the same data preparation steps as the notebook but as a standalone Python script.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
input_data_path = "/tmp/raw_data.csv"
output_data_path = "/tmp/processed_data.csv"
train_test_split_ratio = 0.2
random_seed = 42

def create_sample_data():
    """Create sample dataset for demo purposes."""
    logger.info("Creating sample dataset for demo")
    np.random.seed(random_seed)
    n_samples = 1000
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Save sample data
    os.makedirs(os.path.dirname(input_data_path), exist_ok=True)
    df.to_csv(input_data_path, index=False)
    return df

def load_and_validate_data():
    """Load data and perform basic validation."""
    try:
        if not os.path.exists(input_data_path):
            df = create_sample_data()
        else:
            df = pd.read_csv(input_data_path)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Perform data preprocessing including scaling and encoding."""
    df_processed = df.copy()
    
    # Handle missing values
    for column in df_processed.columns:
        if df_processed[column].dtype == 'object':
            # Fill categorical missing values with mode
            df_processed[column] = df_processed[column].fillna(df_processed[column].mode()[0])
        else:
            # Fill numerical missing values with median
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
    
    # Encode categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != 'target']
    
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le
        logger.info(f"Encoded column: {column}")
    
    # Scale numerical features
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if col != 'target']
    
    scaler = StandardScaler()
    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
    
    logger.info(f"Scaled columns: {list(numerical_columns)}")
    
    # Save preprocessing artifacts
    artifacts_dir = "/tmp/preprocessing_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(scaler, f"{artifacts_dir}/scaler.pkl")
    joblib.dump(label_encoders, f"{artifacts_dir}/label_encoders.pkl")
    
    logger.info("Preprocessing artifacts saved")
    
    return df_processed

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train-test split and save to separate files."""
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save datasets
    output_dir = os.path.dirname(output_data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = f"{output_dir}/train_data.csv"
    test_path = f"{output_dir}/test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Also save the complete processed dataset
    df.to_csv(output_data_path, index=False)
    
    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    logger.info(f"Data saved to: {output_data_path}")
    
    return train_df, test_df

def generate_data_quality_report(df_original, df_processed):
    """Generate a data quality report."""
    report = {
        'original_shape': [int(x) for x in df_original.shape],
        'processed_shape': [int(x) for x in df_processed.shape],
        'features_count': int(len([col for col in df_processed.columns if col != 'target'])),
        'missing_values_original': int(df_original.isnull().sum().sum()),
        'missing_values_processed': int(df_processed.isnull().sum().sum()),
        'categorical_features': int(len(df_original.select_dtypes(include=['object']).columns)),
        'numerical_features': int(len(df_original.select_dtypes(include=[np.number]).columns) - 1)  # Exclude target
    }
    
    # Save report
    report_path = "/tmp/data_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Data quality report saved to: {report_path}")
    
    return report

def main():
    """Main execution function."""
    logger.info("Starting data preparation pipeline component")
    
    try:
        # Load the data
        df = load_and_validate_data()
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Create train-test split
        train_data, test_data = create_train_test_split(df_processed, train_test_split_ratio, random_seed)
        
        # Generate quality report
        quality_report = generate_data_quality_report(df, df_processed)
        
        # Pipeline outputs
        pipeline_outputs = {
            'processed_data': output_data_path,
            'train_data': '/tmp/train_data.csv',
            'test_data': '/tmp/test_data.csv',
            'scaler_artifact': '/tmp/preprocessing_artifacts/scaler.pkl',
            'encoders_artifact': '/tmp/preprocessing_artifacts/label_encoders.pkl',
            'quality_report': '/tmp/data_quality_report.json'
        }
        
        logger.info("Data preparation pipeline component completed successfully!")
        logger.info("Outputs available for next components:")
        for key, path in pipeline_outputs.items():
            exists = os.path.exists(path)
            logger.info(f"  {key}: {'✓' if exists else '✗'} {path}")
        
        print("Data preparation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())