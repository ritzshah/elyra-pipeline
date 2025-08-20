#!/usr/bin/env python3
"""
Fixed Elyra ML Demo Pipeline - Proper Artifact Management
This script creates a KFP v2 pipeline with correct artifact passing between components
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model, Dataset


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def data_preparation_component(
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    preprocessing_artifacts: Output[Artifact],
    data_quality_report: Output[Artifact]
):
    """Data preparation component that loads, cleans, and prepares data for model training."""
    
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
        return pd.DataFrame(data)
    
    # Create sample data
    df = create_sample_data()
    logger.info(f"Data shape: {df.shape}")
    
    # Handle missing values and preprocessing
    df_processed = df.copy()
    
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
    
    # Create train-test split
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_ratio, random_state=random_seed, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save datasets as CSV files
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    logger.info(f"Saved train data to: {train_data.path}")
    logger.info(f"Saved test data to: {test_data.path}")
    
    # Save preprocessing artifacts
    # Create directory structure for artifacts
    artifacts_dir = os.path.dirname(preprocessing_artifacts.path)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save scaler and encoders to the artifacts directory
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    encoders_path = os.path.join(artifacts_dir, "label_encoders.pkl")
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, encoders_path)
    
    # Create artifact metadata file
    artifact_info = {
        "scaler_path": scaler_path,
        "encoders_path": encoders_path,
        "feature_columns": list(X_train.columns),
        "categorical_columns": list(categorical_columns),
        "numerical_columns": list(numerical_columns)
    }
    
    with open(preprocessing_artifacts.path, 'w') as f:
        json.dump(artifact_info, f, indent=2)
    
    logger.info(f"Saved preprocessing artifacts to: {preprocessing_artifacts.path}")
    
    # Generate quality report
    report = {
        'original_shape': [int(x) for x in df.shape],
        'processed_shape': [int(x) for x in df_processed.shape],
        'features_count': int(len([col for col in df_processed.columns if col != 'target'])),
        'missing_values_original': int(df.isnull().sum().sum()),
        'missing_values_processed': int(df_processed.isnull().sum().sum()),
        'train_shape': [int(x) for x in train_df.shape],
        'test_shape': [int(x) for x in test_df.shape]
    }
    
    with open(data_quality_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Data preparation completed successfully!")


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def model_training_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    preprocessing_artifacts: Input[Artifact],
    best_model: Output[Model],
    model_metadata: Output[Artifact],
    model_results: Output[Artifact]
):
    """Model training component that trains multiple models and selects the best one."""
    
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import os
    from datetime import datetime
    import logging
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading train data from: {train_data.path}")
    logger.info(f"Loading test data from: {test_data.path}")
    logger.info(f"Loading preprocessing artifacts from: {preprocessing_artifacts.path}")
    
    # Load data
    train_df = pd.read_csv(train_data.path)
    test_df = pd.read_csv(test_data.path)
    
    # Load preprocessing artifacts info
    with open(preprocessing_artifacts.path, 'r') as f:
        artifact_info = json.load(f)
    
    logger.info(f"Artifact info: {artifact_info}")
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Define models
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.1, 1.0, 10.0]}
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
        }
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for model_name, config in models.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # Perform grid search
            grid_search = GridSearchCV(
                config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model_instance = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model_instance.predict(X_test)
            y_pred_proba = best_model_instance.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
                'best_params': grid_search.best_params_,
                'cv_score_mean': float(grid_search.best_score_)
            }
            
            results[model_name] = metrics
            trained_models[model_name] = best_model_instance
            
            logger.info(f"{model_name} - F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model_instance = trained_models[best_model_name]
    best_score = results[best_model_name]['f1_score']
    
    logger.info(f"Best model: {best_model_name} (F1: {best_score:.4f})")
    
    # Save best model
    joblib.dump(best_model_instance, best_model.path)
    logger.info(f"Saved best model to: {best_model.path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'best_model_score': float(best_score),
        'training_data_shape': [int(x) for x in X_train.shape],
        'test_data_shape': [int(x) for x in X_test.shape],
        'features': list(X_train.columns),
        'random_seed': 42
    }
    
    with open(model_metadata.path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved model metadata to: {model_metadata.path}")
    
    # Save results
    with open(model_results.path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved model results to: {model_results.path}")
    logger.info("Model training completed successfully!")


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def model_evaluation_component(
    best_model: Input[Model],
    model_metadata: Input[Artifact],
    test_data: Input[Dataset],
    evaluation_report: Output[Artifact],
    deployment_status: Output[Artifact]
):
    """Model evaluation component that performs comprehensive model validation."""
    
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import logging
    from datetime import datetime
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from: {best_model.path}")
    logger.info(f"Loading metadata from: {model_metadata.path}")
    logger.info(f"Loading test data from: {test_data.path}")
    
    # Load model and data
    model = joblib.load(best_model.path)
    test_df = pd.read_csv(test_data.path)
    
    with open(model_metadata.path, 'r') as f:
        metadata = json.load(f)
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Evaluating model: {metadata['best_model']}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Performance validation
    performance_threshold = 0.8
    f1_score_val = metrics['f1_score']
    accuracy_val = metrics['accuracy']
    roc_auc_val = metrics['roc_auc']
    
    validation_results = {
        'performance_threshold': performance_threshold,
        'meets_f1_threshold': f1_score_val >= performance_threshold,
        'meets_accuracy_threshold': accuracy_val >= 0.75,
        'meets_auc_threshold': roc_auc_val >= 0.75,
        'overall_validation': False
    }
    
    validation_results['overall_validation'] = (
        validation_results['meets_f1_threshold'] and
        validation_results['meets_accuracy_threshold'] and
        validation_results['meets_auc_threshold']
    )
    
    if validation_results['overall_validation']:
        status = "APPROVED_FOR_PRODUCTION"
        logger.info("Model validation PASSED - approved for production")
    else:
        status = "REQUIRES_IMPROVEMENT"
        logger.warning("Model validation FAILED - requires improvement")
    
    validation_results['deployment_status'] = status
    
    # Generate comprehensive report
    report = {
        'evaluation_summary': {
            'timestamp': datetime.now().isoformat(),
            'model_type': metadata['best_model'],
            'model_score': metadata.get('best_model_score', 0.0),
            'overall_status': status
        },
        'performance_metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'validation_results': validation_results,
        'deployment_recommendation': {
            'ready_for_production': validation_results['overall_validation'],
            'confidence_level': 'HIGH' if validation_results['overall_validation'] else 'LOW',
            'next_steps': [
                "Deploy to KServe" if validation_results['overall_validation'] else "Improve model performance",
                "Set up monitoring",
                "Configure auto-scaling"
            ] if validation_results['overall_validation'] else [
                "Retrain with more data",
                "Feature engineering",
                "Hyperparameter tuning"
            ]
        }
    }
    
    # Save evaluation report
    with open(evaluation_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved evaluation report to: {evaluation_report.path}")
    
    # Save deployment status
    deployment_info = {
        'deployment_approved': validation_results['overall_validation'],
        'deployment_status': status,
        'model_performance': metrics,
        'model_metadata': metadata
    }
    
    with open(deployment_status.path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info(f"Saved deployment status to: {deployment_status.path}")
    logger.info(f"Model evaluation completed. Status: {status}")
    logger.info(f"F1 Score: {f1_score_val:.4f}")
    
    # Print summary for logs
    print("="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    print(f"Model Type: {metadata['best_model']}")
    print(f"F1 Score: {f1_score_val:.4f}")
    print(f"Accuracy: {accuracy_val:.4f}")
    print(f"ROC AUC: {roc_auc_val:.4f}")
    print(f"Deployment Status: {status}")
    print(f"Production Ready: {'Yes' if validation_results['overall_validation'] else 'No'}")
    print("="*50)


@pipeline(
    name="elyra-ml-demo-pipeline-fixed",
    description="Fixed end-to-end ML pipeline with proper artifact management"
)
def elyra_ml_demo_pipeline_fixed():
    """Define the ML pipeline with proper artifact passing between components."""
    
    # Data preparation task
    data_prep_task = data_preparation_component()
    data_prep_task.set_display_name("Data Preparation")
    
    # Model training task - explicitly pass artifacts from data prep
    model_train_task = model_training_component(
        train_data=data_prep_task.outputs['train_data'],
        test_data=data_prep_task.outputs['test_data'],
        preprocessing_artifacts=data_prep_task.outputs['preprocessing_artifacts']
    )
    model_train_task.set_display_name("Model Training")
    model_train_task.after(data_prep_task)
    
    # Model evaluation task - explicitly pass artifacts from both previous tasks
    model_eval_task = model_evaluation_component(
        best_model=model_train_task.outputs['best_model'],
        model_metadata=model_train_task.outputs['model_metadata'],
        test_data=data_prep_task.outputs['test_data']  # Re-use test data from data prep
    )
    model_eval_task.set_display_name("Model Evaluation")
    model_eval_task.after(model_train_task)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=elyra_ml_demo_pipeline_fixed,
        package_path="elyra-ml-demo-pipeline-fixed.yaml"
    )
    print("Fixed pipeline compiled successfully to: elyra-ml-demo-pipeline-fixed.yaml")
    print("This version properly handles artifact passing between components.")
    print("You can now import this file into OpenShift AI Pipeline UI")