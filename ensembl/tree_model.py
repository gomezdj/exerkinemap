# Core Tree Ensemble Model

class ExerkineTreeEnsemble:
    def __init__(self):
        self.models = {
            "xgb": XGBoost,
            "lgbm": LightGBM,
            "rf": RandomForest
        }