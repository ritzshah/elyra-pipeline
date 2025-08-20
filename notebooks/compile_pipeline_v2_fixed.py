#!/usr/bin/env python3
"""
KFP v2 Compatible Elyra ML Demo Pipeline
This script creates a KFP v2 pipeline with proper task naming and artifact references
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Artifact, Model, Dataset


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def data_preparation(
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    artifacts_info: Output[Artifact]
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
    import tempfile
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parameters
    train_test_split_ratio = 0.2
    random_seed = 42
    
    logger.info("Starting data preparation...")
    logger.info(f"Train data output path: {train_data.path}")
    logger.info(f"Test data output path: {test_data.path}")
    logger.info(f"Artifacts info output path: {artifacts_info.path}")
    
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
    logger.info(f"Original data shape: {df.shape}")
    
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
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(train_data.path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data.path), exist_ok=True)
    os.makedirs(os.path.dirname(artifacts_info.path), exist_ok=True)
    
    # Save datasets as CSV files
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    logger.info(f"Saved train data: {train_data.path}")
    logger.info(f"Saved test data: {test_data.path}")
    
    # Create and save preprocessing artifacts info
    artifact_data = {
        "preprocessing_completed": True,
        "feature_columns": list(X_train.columns),
        "categorical_columns": list(categorical_columns),
        "numerical_columns": list(numerical_columns),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "scaler_fitted": True,
        "encoders_fitted": bool(label_encoders),
        "random_seed": random_seed
    }
    
    with open(artifacts_info.path, 'w') as f:
        json.dump(artifact_data, f, indent=2)
    
    logger.info(f"Saved artifacts info: {artifacts_info.path}")
    logger.info("Data preparation completed successfully!")


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def model_training(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    artifacts_info: Input[Artifact],
    best_model: Output[Model],
    model_metrics: Output[Artifact]
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
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model training...")
    logger.info(f"Train data input path: {train_data.path}")
    logger.info(f"Test data input path: {test_data.path}")
    logger.info(f"Artifacts info input path: {artifacts_info.path}")
    logger.info(f"Model output path: {best_model.path}")
    logger.info(f"Metrics output path: {model_metrics.path}")
    
    # Verify input files exist
    if not os.path.exists(train_data.path):
        raise FileNotFoundError(f"Train data not found at: {train_data.path}")
    if not os.path.exists(test_data.path):
        raise FileNotFoundError(f"Test data not found at: {test_data.path}")
    if not os.path.exists(artifacts_info.path):
        raise FileNotFoundError(f"Artifacts info not found at: {artifacts_info.path}")
    
    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_csv(train_data.path)
    test_df = pd.read_csv(test_data.path)
    
    # Load artifacts info
    with open(artifacts_info.path, 'r') as f:
        artifact_data = json.load(f)
    
    logger.info(f"Artifacts info: {artifact_data}")
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Features: {list(X_train.columns)}")
    
    # Define models with simpler parameter grids for faster execution
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.1, 1.0]}
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100]}
        }
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for model_name, config in models.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # Perform grid search with fewer CV folds for speed
            grid_search = GridSearchCV(
                config['model'], config['params'], cv=3, scoring='f1', n_jobs=-1
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
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if not results:
        raise Exception("No models were successfully trained!")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model_instance = trained_models[best_model_name]
    best_score = results[best_model_name]['f1_score']
    
    logger.info(f"Best model: {best_model_name} (F1: {best_score:.4f})")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(best_model.path), exist_ok=True)
    os.makedirs(os.path.dirname(model_metrics.path), exist_ok=True)
    
    # Save best model
    joblib.dump(best_model_instance, best_model.path)
    logger.info(f"Saved best model to: {best_model.path}")
    
    # Save comprehensive metrics
    metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'best_model_name': best_model_name,
        'best_model_score': float(best_score),
        'all_model_results': results,
        'training_data_shape': list(X_train.shape),
        'test_data_shape': list(X_test.shape),
        'features': list(X_train.columns),
        'artifact_data': artifact_data
    }
    
    with open(model_metrics.path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Saved model metrics to: {model_metrics.path}")
    logger.info("Model training completed successfully!")


@component(
    base_image="quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.9-2023b-20231016",
    packages_to_install=["pandas", "scikit-learn", "numpy", "joblib"]
)
def model_evaluation(
    best_model: Input[Model],
    model_metrics: Input[Artifact],
    test_data: Input[Dataset],
    evaluation_results: Output[Artifact]
):
    """Model evaluation component that performs comprehensive model validation."""
    
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import logging
    import os
    from datetime import datetime
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Model input path: {best_model.path}")
    logger.info(f"Metrics input path: {model_metrics.path}")
    logger.info(f"Test data input path: {test_data.path}")
    logger.info(f"Evaluation results output path: {evaluation_results.path}")
    
    # Verify input files exist
    if not os.path.exists(best_model.path):
        raise FileNotFoundError(f"Model not found at: {best_model.path}")
    if not os.path.exists(model_metrics.path):
        raise FileNotFoundError(f"Model metrics not found at: {model_metrics.path}")
    if not os.path.exists(test_data.path):
        raise FileNotFoundError(f"Test data not found at: {test_data.path}")
    
    # Load model and data
    logger.info("Loading model and data...")
    model = joblib.load(best_model.path)
    test_df = pd.read_csv(test_data.path)
    
    with open(model_metrics.path, 'r') as f:
        metrics_data = json.load(f)
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    logger.info(f"Evaluating model: {metrics_data['best_model_name']}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    current_metrics = {
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
    f1_score_val = current_metrics['f1_score']
    accuracy_val = current_metrics['accuracy']
    roc_auc_val = current_metrics['roc_auc']
    
    validation_results = {
        'performance_threshold': performance_threshold,
        'meets_f1_threshold': f1_score_val >= performance_threshold,
        'meets_accuracy_threshold': accuracy_val >= 0.75,
        'meets_auc_threshold': roc_auc_val >= 0.75
    }
    
    validation_results['overall_validation'] = (
        validation_results['meets_f1_threshold'] and
        validation_results['meets_accuracy_threshold'] and
        validation_results['meets_auc_threshold']
    )
    
    status = "APPROVED_FOR_PRODUCTION" if validation_results['overall_validation'] else "REQUIRES_IMPROVEMENT"
    
    logger.info(f"Validation status: {status}")
    logger.info(f"F1 Score: {f1_score_val:.4f}")
    logger.info(f"Accuracy: {accuracy_val:.4f}")
    logger.info(f"ROC AUC: {roc_auc_val:.4f}")
    
    # Generate comprehensive evaluation report
    evaluation_report = {
        'evaluation_summary': {
            'timestamp': datetime.now().isoformat(),
            'model_type': metrics_data['best_model_name'],
            'model_training_score': metrics_data['best_model_score'],
            'evaluation_status': status
        },
        'current_performance_metrics': current_metrics,
        'training_performance_metrics': metrics_data.get('all_model_results', {}),
        'confusion_matrix': cm.tolist(),
        'validation_results': validation_results,
        'deployment_recommendation': {
            'ready_for_production': validation_results['overall_validation'],
            'confidence_level': 'HIGH' if validation_results['overall_validation'] else 'LOW',
            'next_steps': [
                "Deploy to KServe" if validation_results['overall_validation'] else "Improve model performance",
                "Set up monitoring and alerting",
                "Configure auto-scaling"
            ] if validation_results['overall_validation'] else [
                "Retrain with more data",
                "Feature engineering",
                "Hyperparameter tuning",
                "Data quality improvement"
            ]
        },
        'model_details': {
            'training_timestamp': metrics_data.get('timestamp'),
            'features_used': metrics_data.get('features', []),
            'training_data_shape': metrics_data.get('training_data_shape', []),
            'test_data_shape': metrics_data.get('test_data_shape', [])
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(evaluation_results.path), exist_ok=True)
    
    # Save evaluation results
    with open(evaluation_results.path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    logger.info(f"Saved evaluation results to: {evaluation_results.path}")
    
    # Print summary for logs
    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model Type: {metrics_data['best_model_name']}")
    print(f"Training F1 Score: {metrics_data['best_model_score']:.4f}")
    print(f"Evaluation F1 Score: {f1_score_val:.4f}")
    print(f"Accuracy: {accuracy_val:.4f}")
    print(f"ROC AUC: {roc_auc_val:.4f}")
    print(f"Deployment Status: {status}")
    print(f"Production Ready: {'Yes' if validation_results['overall_validation'] else 'No'}")
    if not validation_results['overall_validation']:
        print("\nValidation Failures:")
        if not validation_results['meets_f1_threshold']:
            print(f"  - F1 Score {f1_score_val:.4f} < {performance_threshold}")
        if not validation_results['meets_accuracy_threshold']:
            print(f"  - Accuracy {accuracy_val:.4f} < 0.75")
        if not validation_results['meets_auc_threshold']:
            print(f"  - ROC AUC {roc_auc_val:.4f} < 0.75")
    print("=" * 60)
    
    logger.info("Model evaluation completed successfully!")


@pipeline(
    name="elyra-ml-demo-pipeline-v2",
    description="KFP v2 compatible ML pipeline with proper task naming"
)
def elyra_ml_demo_pipeline_v2():
    """Define the ML pipeline with KFP v2 compatible task naming and artifact passing."""
    
    # Data preparation task - use simple task name
    prep_task = data_preparation()
    prep_task.set_display_name("Data Preparation")
    
    # Model training task - use simple task name and explicit artifact passing
    train_task = model_training(
        train_data=prep_task.outputs['train_data'],
        test_data=prep_task.outputs['test_data'],
        artifacts_info=prep_task.outputs['artifacts_info']
    )
    train_task.set_display_name("Model Training")
    
    # Model evaluation task - use simple task name and explicit artifact passing
    eval_task = model_evaluation(
        best_model=train_task.outputs['best_model'],
        model_metrics=train_task.outputs['model_metrics'],
        test_data=prep_task.outputs['test_data']  # Re-use test data from prep
    )
    eval_task.set_display_name("Model Evaluation")


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=elyra_ml_demo_pipeline_v2,
        package_path="elyra-ml-demo-pipeline-v2.yaml"
    )
    print("KFP v2 compatible pipeline compiled successfully!")
    print("File: elyra-ml-demo-pipeline-v2.yaml")
    print("\nKey fixes:")
    print("- Simple component function names without suffixes")
    print("- Proper artifact type declarations")
    print("- Explicit file existence checks")
    print("- Comprehensive error handling and logging")
    print("- Compatible with KFP v2 task naming conventions")