import os 
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "processed_data"))
model_dir = os.path.abspath(os.path.join(base_dir, "..","..", "models"))
os.makedirs(model_dir, exist_ok=True) 

X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

param_grid = {
    "ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    "lasso": {
        "model": Lasso(max_iter=10000),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0]
        }
    },
    "random_forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "gradient_boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 300],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }   
}

#mse_scorer = make_scorer(
##    mean_squared_error,
 #   greater_is_better=False,
 #   squared=False
#    )

best_overall_model = None
best_overall_score = float("inf")
best_model_name = ""

for name, cfg in param_grid.items():
    grid_search = GridSearchCV(
        estimator=cfg["model"],
        param_grid=cfg["params"],
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        error_score="raise"
    )
    
    grid_search.fit(X_train, y_train)

    rmse = -grid_search.best_score_
    print(f"Best RMSE for {name}: {rmse:.4f} with params: {grid_search.best_params_}")
    if rmse < best_overall_score:
        best_overall_score = rmse
        best_overall_model = grid_search.best_estimator_
        best_model_name = name
        bast_params = grid_search.best_params_

best_params = {
    "model_name": best_model_name,
    "parameters": bast_params
}

model_path = os.path.join(model_dir, f"params.pkl")
joblib.dump(best_params, model_path)
print(f"Best overall model: {best_model_name} with RMSE: {best_overall_score:.4f}")
print(f"Model saved to {model_path}")