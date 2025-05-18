import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

def evaluate_classification_model(model, X, y, cv=5):
    """
    Evaluate a classification model using cross-validation.
    
    Args:
        model: Trained classification model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv (int): Number of cross-validation folds
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    
    # Return metrics
    return {
        'accuracy': cv_accuracy.mean(),
        'accuracy_std': cv_accuracy.std(),
        'precision': cv_precision.mean(),
        'precision_std': cv_precision.std(),
        'recall': cv_recall.mean(),
        'recall_std': cv_recall.std(),
        'f1': cv_f1.mean(),
        'f1_std': cv_f1.std()
    }

def evaluate_regression_model(model, X, y, cv=5):
    """
    Evaluate a regression model using cross-validation.
    
    Args:
        model: Trained regression model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv (int): Number of cross-validation folds
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Cross-validation scores
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
    
    # Return metrics
    return {
        'rmse': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'r2': np.mean(r2_scores),
        'r2_std': np.std(r2_scores)
    }

def evaluate_prediction_calibration(model, X, y, n_bins=10):
    """
    Evaluate the calibration of probability predictions.
    
    Args:
        model: Trained classification model with predict_proba method
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Binary target variable
        n_bins (int): Number of bins for calibration curve
    
    Returns:
        tuple: (prob_true, prob_pred) arrays for plotting calibration curve
    """
    # Get probability predictions
    y_prob = model.predict_proba(X)[:, 1]
    
    # Sort by predicted probability
    sorted_indices = np.argsort(y_prob)
    y_sorted = y.iloc[sorted_indices].values
    y_prob_sorted = y_prob[sorted_indices]
    
    # Create bins
    bin_size = len(y_sorted) // n_bins
    bins = []
    
    # Calculate actual vs predicted probabilities for each bin
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_sorted)
        
        bin_y = y_sorted[start_idx:end_idx]
        bin_prob = y_prob_sorted[start_idx:end_idx].mean()
        
        actual_prob = bin_y.mean()
        predicted_prob = bin_prob
        
        bins.append((actual_prob, predicted_prob))
    
    # Convert to arrays
    prob_true = np.array([b[0] for b in bins])
    prob_pred = np.array([b[1] for b in bins])
    
    return prob_true, prob_pred

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
    
    Returns:
        pd.DataFrame: DataFrame with feature importance scores
    """
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        return importance_df.sort_values('Importance', ascending=False)
    except AttributeError:
        # For models without feature_importances_ attribute
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': [1/len(feature_names)] * len(feature_names)
        })

def evaluate_ensemble_components(ensemble_models, X, y, weights=None):
    """
    Evaluate individual components of an ensemble model.
    
    Args:
        ensemble_models (list): List of component models
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        weights (list, optional): Model weights in ensemble
    
    Returns:
        pd.DataFrame: Performance metrics for each component model
    """
    if weights is None:
        weights = [1/len(ensemble_models)] * len(ensemble_models)
    
    results = []
    
    for i, model in enumerate(ensemble_models):
        # Get model name
        model_name = type(model).__name__
        
        # Evaluate model
        y_pred = model.predict(X)
        
        # Calculate metrics
        if y.dtype == 'object' or y.nunique() <= 5:  # Classification
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            results.append({
                'Model': model_name,
                'Weight': weights[i],
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        else:  # Regression
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            results.append({
                'Model': model_name,
                'Weight': weights[i],
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
    
    return pd.DataFrame(results)
