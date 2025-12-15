import os 
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "processed_data"))
model_dir = os.path.abspath(os.path.join(base_dir, "..","..", "models"))


X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()

payload = joblib.load(os.path.join(model_dir, "params.pkl"))

model_name = payload["model_name"]
model_params = payload["parameters"]


model_factory = {
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor
}  

model = model_factory[model_name](**model_params)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(model_dir, "model.pkl"))