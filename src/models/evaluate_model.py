import os 
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.abspath(os.path.join(base_dir, "..","..", "metrics"))
model_dir = os.path.abspath(os.path.join(base_dir, "..","..", "models"))
data_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "processed_data"))
output_dir  = os.path.abspath(os.path.join(base_dir, "..","..", "data"))

model = joblib.load(os.path.join(model_dir, "model.pkl"))

X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})

predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

with open(os.path.join(metrics_dir, "evaluation_metrics.json"), "w") as f:
    json.dump({"mean_squared_error": mse, "r2_score": r2}, f)

