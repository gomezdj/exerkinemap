import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

class ExerkineTreeEnsemble:

    def __init__(self):
        self.models = {}
        self.weights = {}

    def train(self, X, y):

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )

        xgbr = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6
        )

        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05
        )

        self.models = {
            "rf": rf,
            "xgb": xgbr,
            "lgbm": lgbm
        }

        scores = {}
        for name, model in self.models.items():
            cv = cross_val_score(model, X, y, cv=5, scoring="r2")
            scores[name] = np.mean(cv)
            model.fit(X, y)

        total = sum(scores.values())
        self.weights = {k: v/total for k, v in scores.items()}

    def predict(self, X):

        preds = 0
        for name, model in self.models.items():
            preds += self.weights[name] * model.predict(X)

        return preds