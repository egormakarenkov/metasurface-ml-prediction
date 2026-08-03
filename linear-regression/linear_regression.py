import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load Cleaned Training Data (No headers, no empty columns)
train_data = pd.read_excel('New_data_02_train_cleaned.xlsx', header=None)
X_train = train_data.iloc[:, 0:6]  # Cols 0-5 are features
R_train = train_data.iloc[:, 6]    # Col 6 is Reflectance
T_train = train_data.iloc[:, 7]    # Col 7 is Transmittance

# Train Models
model_R = LinearRegression().fit(X_train, R_train)
model_T = LinearRegression().fit(X_train, T_train)

# 2. Load Cleaned Testing Data (The 10 extracted lines)
test_data = pd.read_excel('New_data_02_test_cleaned.xlsx', header=None)
X_test = test_data.iloc[:, 0:6]
R_test_actual = test_data.iloc[:, 6]
T_test_actual = test_data.iloc[:, 7]

# Predict on the 10 lines
R_pred = model_R.predict(X_test)
T_pred = model_T.predict(X_test)

# Calculate Average MSE
mse_R_10 = mean_squared_error(R_test_actual, R_pred)
mse_T_10 = mean_squared_error(T_test_actual, T_pred)

print(f"--- LINEAR REGRESSION TEST ON 10 UNSEEN LINES ---")
print(f"Average MSE for Reflectance: {mse_R_10:.6f}")
print(f"Average MSE for Transmittance: {mse_T_10:.6f}")
