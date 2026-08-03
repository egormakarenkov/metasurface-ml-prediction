import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Cleaned Training Data
train_data = pd.read_excel('New_data_02_train_cleaned.xlsx', header=None)
X_train = train_data.iloc[:, 0:6]
yR_train = train_data.iloc[:, 6]
yT_train = train_data.iloc[:, 7]

# Train Models
xgb_R = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=42)
xgb_T = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=42)

xgb_R.fit(X_train, yR_train)
xgb_T.fit(X_train, yT_train)

# 2. Load Cleaned Testing Data (10 lines)
test_data = pd.read_excel('New_data_02_test_cleaned.xlsx', header=None)
X_test = test_data.iloc[:, 0:6]
yR_test_actual = test_data.iloc[:, 6]
yT_test_actual = test_data.iloc[:, 7]

# Predict and Evaluate
pred_R = xgb_R.predict(X_test)
pred_T = xgb_T.predict(X_test)

mse_R = mean_squared_error(yR_test_actual, pred_R)
mse_T = mean_squared_error(yT_test_actual, pred_T)

print(f"--- XGBOOST TEST ON 10 UNSEEN LINES ---")
print(f"Average MSE for Reflectance: {mse_R:.6f}")
print(f"Average MSE for Transmittance: {mse_T:.6f}")
