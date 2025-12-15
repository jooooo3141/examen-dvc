import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "processed_data"))

X_train_path = os.path.join(data_dir, "X_train.csv")
X_test_path = os.path.join(data_dir, "X_test.csv")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled_df.to_csv(os.path.join(data_dir, "X_train_scaled.csv"), index=False)
X_test_scaled_df.to_csv(os.path.join(data_dir, "X_test_scaled.csv"), index=False)
print("Scaled datasets saved to processed data directory.")