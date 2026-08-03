import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

print("Loading data and training models... this might take a minute...")

# 1. Load the NEW 1000-split Cleaned Data
train_data = pd.read_excel('New_data_02_train_1000.xlsx', header=None)
X_train = train_data.iloc[:, 0:6]
yR_train = train_data.iloc[:, 6]
yT_train = train_data.iloc[:, 7]

test_data = pd.read_excel('New_data_02_test_1000.xlsx', header=None)
X_test = test_data.iloc[:, 0:6]

# Rename columns so the SHAP graphs look professional in the final paper
feature_names = ['Frequency (GHz)', 'C_V', 'C_V_bot', 'L_V', 'R_V', 'Theta']
X_train.columns = feature_names
X_test.columns = feature_names

# 2. Re-train the winning Random Forest models on the 8,940 rows
rf_R = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42)
rf_T = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42)

rf_R.fit(X_train, yR_train)
rf_T.fit(X_train, yT_train)

print("Models trained! Running SHAP Explainers on the 1000 test points (cracking the black box)...")

# 3. Initialize SHAP TreeExplainer and run on the full 1000-row test set
explainer_R = shap.TreeExplainer(rf_R)
shap_values_R = explainer_R.shap_values(X_test)

explainer_T = shap.TreeExplainer(rf_T)
shap_values_T = explainer_T.shap_values(X_test)

# 4. Generate and Save Summary Plot for Reflectance
plt.figure(figsize=(8, 6))
plt.title("SHAP Feature Importance: Reflectance (R)", fontsize=14, pad=20)
shap.summary_plot(shap_values_R, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_R_1000.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Generate and Save Summary Plot for Transmittance
plt.figure(figsize=(8, 6))
plt.title("SHAP Feature Importance: Transmittance (T)", fontsize=14, pad=20)
shap.summary_plot(shap_values_T, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary_T_1000.png', dpi=300, bbox_inches='tight')
plt.close()

print("SHAP analysis complete!")
print("Check your project folder for 'shap_summary_R_1000.png' and 'shap_summary_T_1000.png'.")
