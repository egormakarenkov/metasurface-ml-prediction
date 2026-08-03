import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Cleaned Training Data
train_data = pd.read_excel('New_data_02_train_cleaned.xlsx', header=None)
X_train = train_data.iloc[:, 0:6]
y_train = train_data.iloc[:, 6:8]

# Train Model
rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)

# 2. Load Cleaned Testing Data (10 lines)
test_data = pd.read_excel('New_data_02_test_cleaned.xlsx', header=None)
X_test = test_data.iloc[:, 0:6]
y_test_actual = test_data.iloc[:, 6:8]

# Predict and Evaluate
y_pred = rf.predict(X_test)
mse_R = mean_squared_error(y_test_actual.iloc[:, 0], y_pred[:, 0])
mse_T = mean_squared_error(y_test_actual.iloc[:, 1], y_pred[:, 1])

print(f"--- RANDOM FOREST TEST ON 10 UNSEEN LINES ---")
print(f"Average MSE for Reflectance: {mse_R:.6f}")
print(f"Average MSE for Transmittance: {mse_T:.6f}")
