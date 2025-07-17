import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from string_manipulation import make_func
from data_process_fun import compute_metrics
from typing import Callable, List, Optional, Tuple

class FastLinearRegression:
    """Fast linear regression using closed-form solution (X^T X)^-1 X^T y"""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        self.fitted_ = False
        
    def fit(self, X, y):
        """Fit using closed-form solution for speed"""
        try:
            # Convert to numpy arrays and validate
            X = np.asarray(X)
            y = np.asarray(y)
            
            # Check for NaN or infinity values
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError("Input contains NaN or infinity values")
            
            # Check for very large values that might cause numerical issues
            if np.abs(X).max() > 1e10 or np.abs(y).max() > 1e10:
                raise ValueError("Input contains very large values")
            
            # Add intercept column
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            
            # Closed-form solution: theta = (X^T X)^-1 X^T y
            # Using numpy's lstsq for numerical stability
            params, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            
            # Check if the solution is well-conditioned
            if rank < X_with_intercept.shape[1]:
                raise ValueError("Matrix is rank deficient")
            
            self.intercept_ = params[0]
            self.coef_ = params[1:] if len(params) > 1 else np.array([])
            self.fitted_ = True
            
        except Exception as e:
            # Fallback to sklearn if numerical issues
            try:
                fallback_model = LinearRegression()
                fallback_model.fit(X, y)
                self.coef_ = fallback_model.coef_
                self.intercept_ = fallback_model.intercept_
                self.fitted_ = True
            except Exception as sklearn_error:
                # If even sklearn fails, set to zero model
                self.coef_ = np.zeros(X.shape[1])
                self.intercept_ = np.mean(y) if len(y) > 0 else 0.0
                self.fitted_ = True
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted_:
            raise ValueError("Model not fitted yet")
        X = np.asarray(X)
        
        # Check for NaN/infinity in input
        if not np.isfinite(X).all():
            # Replace non-finite values with zeros
            X = np.where(np.isfinite(X), X, 0.0)
        
        try:
            pred = X @ self.coef_ + self.intercept_
            
            # Check for NaN/infinity in output
            if not np.isfinite(pred).all():
                # Return zeros if prediction contains NaN/infinity
                return np.zeros_like(pred)
                
            return pred
        except Exception:
            # If prediction fails for any reason, return zeros
            return np.zeros(X.shape[0])

class LinearBoostModel:
    """Model 1: Greedy sequential orthogonalized linear regression (FWL-style)"""
    def __init__(self, base_features, poly_degree=3):
        self.base_features = base_features
        self.poly_degree = poly_degree
        self.selected_features = []
        self.feature_functions = {}
        self.feature_bodies = {}
        self.residual_models = []  # Store (feature_name, residual_model) pairs for sequential selection
        self.feature_predictors = []  # Store (feature_name, feature_predictor) for orthogonalization consistency
        self.base_model = None
        self.poly = None
        self.poly_feature_names = []
        self.current_accuracy = 0.0
        
        # Performance optimization: Cache frequently computed matrices
        self._poly_features_cache = None
        self._current_features_cache = None
        self._cache_valid = False
        self._last_cache_data_shape = None
        
    def fit_initial(self, X_raw, y):
        """Fit initial polynomial model"""
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        poly_features = self.poly.fit_transform(X_raw[self.base_features])
        self.poly_feature_names = list(self.poly.get_feature_names_out(self.base_features))
        
        self.base_model = FastLinearRegression()
        self.base_model.fit(poly_features, y)
        
    def evaluate_feature(self, feature_name, feature_body, X_raw, y, logger):
        """Evaluate adding a new feature using greedy sequential orthogonalized selection (FWL-style)"""
        try:
            # Generate feature
            func = make_func(feature_body)
            X_temp = X_raw.copy()
            
            # Apply function safely
            try:
                new_feature = X_temp.apply(func, axis=1)
            except Exception as e:
                logger.warning(f"Feature {feature_name} failed to compute: {e}, skipping")
                return -1, None, None
            
            X_temp[feature_name] = new_feature
            
            # Vectorized feature validation (optimized)
            if not self._validate_feature_vectorized(new_feature, feature_name, logger):
                return -1, None, None
            
            # 1. Get current model prediction using cached method (MAJOR OPTIMIZATION)
            current_pred = self._predict_incremental(X_raw)
            
            # Validate current prediction
            if not np.isfinite(current_pred).all():
                logger.warning(f"Current model prediction contains non-finite values, skipping feature {feature_name}")
                return -1, None, None
                
            residual = y - current_pred
            
            # Validate residual
            if not np.isfinite(residual).all():
                logger.warning(f"Residual contains non-finite values, skipping feature {feature_name}")
                return -1, None, None
            
            # 2. Use cached current features matrix (MAJOR OPTIMIZATION)
            current_features = self._get_current_features_matrix(X_raw)
            
            # Validate combined feature matrix
            if not np.isfinite(current_features).all():
                logger.warning(f"Combined feature matrix contains non-finite values, skipping feature {feature_name}")
                return -1, None, None
            
            # 3. Orthogonalize new feature against existing features (FWL step)
            feature_predictor = None
            if current_features.shape[1] > 0:
                try:
                    feature_predictor = FastLinearRegression()
                    feature_predictor.fit(current_features, X_temp[feature_name])
                    feature_pred = feature_predictor.predict(current_features)
                    
                    # Validate feature prediction
                    if not np.isfinite(feature_pred).all():
                        logger.warning(f"Feature orthogonalization prediction contains non-finite values, skipping feature {feature_name}")
                        return -1, None, None
                        
                    feature_residual = X_temp[feature_name] - feature_pred
                    
                    # Validate feature residual
                    if not np.isfinite(feature_residual).all():
                        logger.warning(f"Feature orthogonalization residual contains non-finite values, skipping feature {feature_name}")
                        return -1, None, None
                        
                except Exception as e:
                    logger.warning(f"Feature orthogonalization failed for {feature_name}: {e}, skipping")
                    return -1, None, None
            else:
                feature_residual = X_temp[feature_name].values
            
            # Check feature residual variance
            if np.var(feature_residual) < 1e-15:
                logger.warning(f"Orthogonalized feature {feature_name} has zero variance, skipping")
                return -1, None, None
            
            # 4. Fit orthogonalized feature against target residual
            try:
                residual_model = FastLinearRegression()
                feature_residual_2d = feature_residual.values.reshape(-1, 1) if hasattr(feature_residual, 'values') else feature_residual.reshape(-1, 1)
                
                # Final validation before fitting
                if not np.isfinite(feature_residual_2d).all():
                    logger.warning(f"Final feature residual contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
                residual_model.fit(feature_residual_2d, residual)
                
                # 5. Compute final prediction
                residual_pred = residual_model.predict(feature_residual_2d)
                
                # Validate residual prediction
                if not np.isfinite(residual_pred).all():
                    logger.warning(f"Residual model prediction contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
                y_pred = current_pred + residual_pred
                
                # Validate final prediction
                if not np.isfinite(y_pred).all():
                    logger.warning(f"Final prediction contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
            except Exception as e:
                logger.warning(f"Residual model fitting failed for {feature_name}: {e}, skipping")
                return -1, None, None
            
            _, accuracy, _ = compute_metrics(y_pred, y, logger)
            return accuracy, residual_model, feature_predictor
            
        except Exception as e:
            logger.error(f"Error evaluating orthogonalized feature {feature_name}: {e}")
            return -1, None, None
    
    def add_feature(self, feature_name, feature_body, residual_model, feature_predictor):
        """Add a feature to the model using orthogonalized residual approach"""
        self.selected_features.append(feature_name)
        self.feature_bodies[feature_name] = feature_body
        self.feature_functions[feature_name] = make_func(feature_body)
        
        # Store the residual model and the feature predictor for this feature
        self.residual_models.append((feature_name, residual_model))
        self.feature_predictors.append((feature_name, feature_predictor))
        
        # Invalidate cache when model changes (OPTIMIZATION)
        self._invalidate_cache()
        # Don't compute accuracy here - it will be computed externally and set via current_accuracy
        
    def set_current_accuracy(self, accuracy):
        """Set the current accuracy of the model"""
        self.current_accuracy = accuracy
        
    def predict(self, X_raw):
        """Make predictions using greedy sequential orthogonalized approach"""
        if self.poly is None:
            return np.zeros(len(X_raw))
            
        # Create a copy to avoid modifying the original
        X_temp = X_raw.copy()
        
        # Start with polynomial features prediction
        poly_features = self.poly.transform(X_temp[self.base_features])
        
        if not hasattr(self, 'base_model') or self.base_model is None:
            # Initialize base model if needed
            self.base_model = FastLinearRegression()
            # Fit it on polynomial features with zero target (will be updated when add_feature is called)
            self.base_model.fit(poly_features, np.zeros(len(X_temp)))
        
        current_pred = self.base_model.predict(poly_features)
        current_features = poly_features.copy() # type: ignore
        
        # Add features sequentially with orthogonalization using stored predictors
        for i, (fname, residual_model) in enumerate(self.residual_models):
            # Compute feature value
            if fname in X_temp.columns:
                feature_value = X_temp[fname].values
            else:
                feature_value = X_temp.apply(self.feature_functions[fname], axis=1).values
            
            # Orthogonalize using stored predictor (consistent with training)
            if i < len(self.feature_predictors) and self.feature_predictors[i][1] is not None:
                feature_predictor = self.feature_predictors[i][1]
                feature_pred = feature_predictor.predict(current_features)
                feature_residual = feature_value - feature_pred
            else:
                # If no predictor stored (shouldn't happen but defensive), use raw feature
                feature_residual = feature_value
            
            # Add residual prediction
            residual_pred = residual_model.predict(feature_residual.values.reshape(-1, 1) if hasattr(feature_residual, 'values') else feature_residual.reshape(-1, 1))
            current_pred += residual_pred
            
            # Update current features for next iteration
            current_features = np.column_stack([current_features, feature_value])
        
        return current_pred
    
    def get_coefficients(self):
        """Get model coefficients for orthogonalized sequential model"""
        coef_dict = {}
        
        if self.base_model is not None:
            # Polynomial coefficients
            for i, name in enumerate(self.poly_feature_names):
                coef_dict[f"poly_{name}"] = self.base_model.coef_[i] # type: ignore
            coef_dict['poly_intercept'] = self.base_model.intercept_
        
        # Residual model coefficients
        for fname, residual_model in self.residual_models:
            coef_dict[f"residual_{fname}"] = residual_model.coef_[0]
            coef_dict[f"residual_{fname}_intercept"] = residual_model.intercept_
            
        return coef_dict
    
    def _invalidate_cache(self):
        """Invalidate cached matrices when model changes"""
        self._cache_valid = False
        self._poly_features_cache = None
        self._current_features_cache = None
        
    def _validate_feature_vectorized(self, feature_values, feature_name, logger):
        """Vectorized feature validation (shared with ResidualBoostingModel)"""
        mask = (
            pd.notna(feature_values) &
            np.isfinite(feature_values) &
            (np.abs(feature_values) <= 1e10)
        )
        
        if not mask.all():
            valid_ratio = mask.mean()
            if valid_ratio < 0.95:
                logger.warning(f"Feature {feature_name} has {(1-valid_ratio)*100:.1f}% invalid values, skipping")
                return False
            else:
                median_val = feature_values[mask].median() if mask.any() else 0.0
                feature_values[~mask] = median_val
        
        if feature_values.var() < 1e-15:
            logger.warning(f"Feature {feature_name} has zero variance, skipping")
            return False
            
        return True
    
    def _get_poly_features(self, X_raw, force_rebuild=False):
        """Get polynomial features with caching"""
        if (self._poly_features_cache is None or force_rebuild or 
            not self._cache_valid or X_raw.shape != self._last_cache_data_shape):
            self._poly_features_cache = self.poly.transform(X_raw[self.base_features]) # type: ignore
            self._last_cache_data_shape = X_raw.shape
        return self._poly_features_cache
    
    def _get_current_features_matrix(self, X_raw, force_rebuild=False):
        """Get current feature matrix (poly + selected features) with caching"""
        if (self._current_features_cache is None or force_rebuild or 
            not self._cache_valid or X_raw.shape != self._last_cache_data_shape):
            
            poly_features = self._get_poly_features(X_raw, force_rebuild)
            current_features = poly_features.copy() # type: ignore
            
            # Add selected features in batch (vectorized)
            if self.selected_features:
                selected_feature_values = []
                for fname in self.selected_features:
                    if fname in X_raw.columns:
                        selected_feature_values.append(X_raw[fname].values)
                    else:
                        # Compute feature using stored function
                        feature_vals = X_raw.apply(self.feature_functions[fname], axis=1).values
                        selected_feature_values.append(feature_vals)
                
                if selected_feature_values:
                    selected_feature_matrix = np.column_stack(selected_feature_values)
                    current_features = np.column_stack([current_features, selected_feature_matrix])
            
            self._current_features_cache = current_features
            self._cache_valid = True
            
        return self._current_features_cache
    
    def _predict_base_only(self, X_raw):
        """Predict using only polynomial features (cached)"""
        if self.base_model is None:
            return np.zeros(len(X_raw))
        poly_features = self._get_poly_features(X_raw)
        return self.base_model.predict(poly_features)
    
    def _predict_incremental(self, X_raw):
        """Incremental prediction using cached base + residual models"""
        # Start with cached base prediction
        current_pred = self._predict_base_only(X_raw)
        
        # Add residual predictions incrementally (reuse cached features)
        if self.residual_models:
            current_features = self._get_poly_features(X_raw)
            
            for i, (fname, residual_model) in enumerate(self.residual_models):
                # Compute feature value
                if fname in X_raw.columns:
                    feature_value = X_raw[fname].values
                else:
                    feature_value = X_raw.apply(self.feature_functions[fname], axis=1).values
                
                # Orthogonalize using stored predictor
                if i < len(self.feature_predictors) and self.feature_predictors[i][1] is not None:
                    feature_predictor = self.feature_predictors[i][1]
                    # Build features matrix incrementally for prediction
                    features_for_pred = current_features.copy() # type: ignore
                    for j in range(i):
                        prev_fname = self.residual_models[j][0]
                        if prev_fname in X_raw.columns:
                            prev_feature = X_raw[prev_fname].values
                        else:
                            prev_feature = X_raw.apply(self.feature_functions[prev_fname], axis=1).values
                        features_for_pred = np.column_stack([features_for_pred, prev_feature])
                    
                    feature_pred = feature_predictor.predict(features_for_pred)
                    feature_residual = feature_value - feature_pred
                else:
                    feature_residual = feature_value
                
                # Add residual prediction
                residual_pred = residual_model.predict(feature_residual.reshape(-1, 1))
                current_pred += residual_pred
        
        return current_pred
    
    def batch_evaluate_features(self, feature_list, X_raw, y, logger):
        """Batch evaluate multiple features for efficiency"""
        results = []
        
        # Precompute base prediction once for all features
        base_pred = self._predict_base_only(X_raw)
        current_features = self._get_current_features_matrix(X_raw)
        
        for feature_info in feature_list:
            feature_name, feature_body = feature_info
            accuracy, residual_model, feature_predictor = self.evaluate_feature_with_precomputed(
                feature_name, feature_body, X_raw, y, logger, base_pred, current_features)
            if accuracy > 0:
                results.append((feature_name, feature_body, accuracy, residual_model, feature_predictor))
        
        return results
    
    def evaluate_feature_with_precomputed(self, feature_name, feature_body, X_raw, y, logger, base_pred, current_features):
        """Evaluate feature with precomputed base prediction and features matrix"""
        try:
            # Generate feature
            func = make_func(feature_body)
            X_temp = X_raw.copy()
            
            # Apply function safely
            try:
                new_feature = X_temp.apply(func, axis=1)
            except Exception as e:
                logger.warning(f"Feature {feature_name} failed to compute: {e}, skipping")
                return -1, None, None
            
            X_temp[feature_name] = new_feature
            
            # Vectorized feature validation (optimized)
            if not self._validate_feature_vectorized(new_feature, feature_name, logger):
                return -1, None, None
            
            # Use precomputed base prediction
            current_pred = base_pred.copy()
            
            # Add residual predictions incrementally using precomputed current features
            for i, (fname, residual_model) in enumerate(self.residual_models):
                if fname in X_raw.columns:
                    feature_value = X_raw[fname].values
                else:
                    feature_value = X_raw.apply(self.feature_functions[fname], axis=1).values
                
                # Use precomputed features for orthogonalization
                if i < len(self.feature_predictors) and self.feature_predictors[i][1] is not None:
                    feature_predictor = self.feature_predictors[i][1]
                    # Build features matrix up to this point
                    features_for_pred = current_features[:, :current_features.shape[1]-(len(self.residual_models)-i-1)] if i > 0 else current_features[:, :len(self.poly_feature_names)]
                    feature_pred = feature_predictor.predict(features_for_pred)
                    feature_residual = feature_value - feature_pred
                else:
                    feature_residual = feature_value
                
                residual_pred = residual_model.predict(feature_residual.reshape(-1, 1))
                current_pred += residual_pred
            
            # Validate current prediction
            if not np.isfinite(current_pred).all():
                logger.warning(f"Current model prediction contains non-finite values, skipping feature {feature_name}")
                return -1, None, None
                
            residual = y - current_pred
            
            # Validate residual
            if not np.isfinite(residual).all():
                logger.warning(f"Residual contains non-finite values, skipping feature {feature_name}")
                return -1, None, None
            
            # Orthogonalize new feature against existing features (FWL step)
            feature_predictor = None
            if current_features.shape[1] > 0:
                try:
                    feature_predictor = FastLinearRegression()
                    feature_predictor.fit(current_features, X_temp[feature_name])
                    feature_pred = feature_predictor.predict(current_features)
                    
                    if not np.isfinite(feature_pred).all():
                        logger.warning(f"Feature orthogonalization prediction contains non-finite values, skipping feature {feature_name}")
                        return -1, None, None
                        
                    feature_residual = X_temp[feature_name] - feature_pred
                    
                    if not np.isfinite(feature_residual).all():
                        logger.warning(f"Feature orthogonalization residual contains non-finite values, skipping feature {feature_name}")
                        return -1, None, None
                        
                except Exception as e:
                    logger.warning(f"Feature orthogonalization failed for {feature_name}: {e}, skipping")
                    return -1, None, None
            else:
                feature_residual = X_temp[feature_name].values
            
            # Check feature residual variance
            if np.var(feature_residual) < 1e-15:
                logger.warning(f"Orthogonalized feature {feature_name} has zero variance, skipping")
                return -1, None, None
            
            # Fit orthogonalized feature against target residual
            try:
                residual_model = FastLinearRegression()
                feature_residual_2d = feature_residual.values.reshape(-1, 1) if hasattr(feature_residual, 'values') else feature_residual.reshape(-1, 1)
                
                if not np.isfinite(feature_residual_2d).all():
                    logger.warning(f"Final feature residual contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
                residual_model.fit(feature_residual_2d, residual)
                residual_pred = residual_model.predict(feature_residual_2d)
                
                if not np.isfinite(residual_pred).all():
                    logger.warning(f"Residual model prediction contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
                y_pred = current_pred + residual_pred
                
                if not np.isfinite(y_pred).all():
                    logger.warning(f"Final prediction contains non-finite values, skipping feature {feature_name}")
                    return -1, None, None
                    
            except Exception as e:
                logger.warning(f"Residual model fitting failed for {feature_name}: {e}, skipping")
                return -1, None, None
            
            _, accuracy, _ = compute_metrics(y_pred, y, logger)
            return accuracy, residual_model, feature_predictor
            
        except Exception as e:
            logger.error(f"Error evaluating orthogonalized feature {feature_name}: {e}")
            return -1, None, None