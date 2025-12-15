import os 
import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "raw.csv"
target = "silica_concentrate"
test_size = 0.2
random_state = 42

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "raw_data"))
output_dir = os.path.abspath(os.path.join(base_dir, "..","..", "data", "processed_data"))
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(os.path.abspath(os.path.join(data_dir, input_file)))

X = df.drop(columns=["date",target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)    

print("Datasets saved to processed data directory.")