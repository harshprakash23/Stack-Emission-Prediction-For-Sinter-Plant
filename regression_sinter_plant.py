import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv('INTERNSHIP_DATA.csv')

# Features and target
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']]
y = df['output']

# Handling missing values (if any)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Training and evaluation
results = {}
for name, model in models.items():
    # Training
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'CV RMSE': cv_rmse}
    
    # Feature importance for tree-based models
    # Feature importance or coefficients for all models
for name, model in models.items():
    if name == 'Linear Regression':
        # Use coefficients for Linear Regression
        feature_importance = pd.Series(model.coef_, index=X.columns)
        feature_importance = feature_importance.abs()  # Take absolute values for visualization
        plt.figure(figsize=(10, 5))
        feature_importance.sort_values().plot(kind='barh')
        plt.title(f'Feature Coefficients - {name}')
        plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
        plt.close()
    elif name in ['Random Forest', 'XGBoost']:
        # Use feature importances for tree-based models
        plt.figure(figsize=(10, 5))
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance.sort_values().plot(kind='barh')
        plt.title(f'Feature Importance - {name}')
        plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
        plt.close()

# Printing results
print("Model Performance:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"CV RMSE: {metrics['CV RMSE']:.4f}")

# Saving results to a DataFrame
results_df = pd.DataFrame(results).T
results_df.to_csv('regression_results.csv')