# Metasurface Reflectance & Transmittance Prediction

Machine learning models that predict the reflectance (R) and transmittance (T) of electromagnetic Rasorber metasurfaces from their design parameters: frequency, top/bottom layer capacitance (C_V, C_V_bot), inductance (L_V), resistance (R_V), and incidence angle (theta).

This started as an independent research project with Sakib Reza's lab at UT Dallas, and is now being developed toward a formal paper. Four modeling approaches are implemented and compared, followed by an explainability pass to connect the model's predictions back to the underlying physics.

## Methods

| Method | Folder | Description |
|---|---|---|
| Multiple Linear Regression | linear-regression/ | Baseline linear model fit with scikit-learn (LinearRegression). |
| Deep Neural Network | neural-network/ | Keras/TensorFlow feedforward network (3 hidden layers, ReLU, dropout) trained to jointly predict R and T. |
| Random Forest Regression | random-forest/ | 300-tree ensemble (RandomForestRegressor); best-performing model. |
| XGBoost | xgboost/ | Gradient-boosted trees (XGBRegressor) as a second ensemble baseline. |
| SHAP Analysis | shap-analysis/ | Explainable AI pass on the Random Forest model using SHAP (Shapley values) to identify which physical parameters drive R and T. |

## Results

Evaluated on an independent 1,000-sample held-out test set (Mean Squared Error, lower is better):

| Model | MSE Reflectance (R) | MSE Transmittance (T) |
|---|---|---|
| Deep Neural Network | 0.096624 | 0.068028 |
| Linear Regression | 0.071659 | 0.053361 |
| XGBoost | 0.001766 | 0.000750 |
| Random Forest | 0.000064 | 0.000023 |

Random Forest performed best by a wide margin. Because the underlying simulation data is generated from discrete, stepped parameter sweeps, tree-based methods (which naturally split on discontinuities) outperform the linear and neural-network baselines, which assume smoother relationships.

### SHAP findings

SHAP analysis on the Random Forest model (see shap-analysis/results/) shows that while frequency and incidence angle (theta) govern where resonance occurs, the geometric resistance parameter (R_V) is the dominant driver of the amplitude of reflection and transmission, consistent with the impedance-matching/dissipation behavior expected of a Rasorber-type absorptive metasurface.

## Data

The training/test datasets used by these scripts (New_data_02*.xlsx) are simulation outputs (HFSS/CST) provided by the research group and are not included in this repository. Each script expects a cleaned Excel file in its own folder with columns:

[Frequency, C_V, C_V_bot, L_V, R_V, theta, Reflectance, Transmittance]

(no header row; see the individual scripts for exact column indices).

## Running the code

Each folder is self-contained. From inside a given method's folder:

pip install pandas scikit-learn xgboost tensorflow shap matplotlib
python script_name.py

## Background

This project has been developed since 2023 in collaboration with Sakib Reza's research group at UT Dallas, and is currently being written up as a formal paper with Lucas Capone and Erik Pineda-Alvarez.

## Author

Egor Makarenkov
