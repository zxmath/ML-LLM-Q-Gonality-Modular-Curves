import xgboost as xgb 
import lightgbm as lgb
def train_xgb_model(X_train, y_train, params):
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_lgbm_model(X_train, y_train, params):
    lgbm_model = lgb.LGBMRegressor(**params)
    lgbm_model.fit(X_train, y_train)
    return lgbm_model
