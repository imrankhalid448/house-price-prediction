# final_script.py
# Standalone script for the House Price Prediction project
# This script loads the Ames Housing dataset, preprocesses the data, trains a stacked ensemble of XGBoost, LightGBM, and CatBoost with RFE,
# generates predictions for the Kaggle test set, and saves the submission file and model artifacts.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Define base directory for the project
BASE_DIR = 'E:\\Imrankhalid\\Basic Libraries\\Initial_Level_Projects\\house_price_prediction_project'

# Step 1: Load and preprocess the training set
print("Loading and preprocessing the training set...")
train_df = pd.read_csv(f'{BASE_DIR}\\dataset\\train.csv')
y_train = train_df['SalePrice']
y_train = np.log1p(y_train)  # Log-transform the target
train_df = train_df.drop(columns=['Id', 'SalePrice'])  # Drop Id and target

# Handle missing values
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].median())
categorical_cols = train_df.select_dtypes(include=['object']).columns
train_df[categorical_cols] = train_df[categorical_cols].fillna(train_df[categorical_cols].mode().iloc[0])

# Feature engineering
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
train_df['HouseAge'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['OverallQual_SF'] = train_df['OverallQual'] * train_df['TotalSF']
train_df['OverallQual_GrLivArea'] = train_df['OverallQual'] * train_df['GrLivArea']
# Add polynomial features
train_df['GrLivArea_Squared'] = train_df['GrLivArea'] ** 2
train_df['TotalSF_Squared'] = train_df['TotalSF'] ** 2
# Add new features
train_df['TotalBathrooms'] = train_df['FullBath'] + 0.5 * train_df['HalfBath'] + train_df['BsmtFullBath'] + 0.5 * train_df['BsmtHalfBath']
train_df['OverallQual_GarageArea'] = train_df['OverallQual'] * train_df['GarageArea']
# Add interaction with Neighborhood (example: OverallQual_Neighborhood)
neighborhood_qual = pd.get_dummies(train_df['Neighborhood'], prefix='Neighborhood')
interaction_terms = pd.DataFrame(index=train_df.index)
for col in neighborhood_qual.columns:
    interaction_terms[f'{col}_OverallQual'] = neighborhood_qual[col] * train_df['OverallQual']
train_df = pd.concat([train_df, interaction_terms], axis=1)
# Add new interaction term
train_df['OverallQual_YearBuilt'] = train_df['OverallQual'] * train_df['YearBuilt']
# Add spatial clustering of neighborhoods
neighborhood_encoded = pd.get_dummies(train_df['Neighborhood']).values
kmeans = KMeans(n_clusters=5, random_state=42)
train_df['Neighborhood_Cluster'] = kmeans.fit_predict(neighborhood_encoded)

# Cap features to prevent extreme predictions (further increased caps)
train_df['GrLivArea'] = train_df['GrLivArea'].clip(upper=6000)  # Increased to 6,000
train_df['TotalSF'] = train_df['TotalSF'].clip(upper=12000)  # Increased to 12,000
train_df['OverallQual_SF'] = train_df['OverallQual_SF'].clip(upper=80000)  # Increased to 80,000
train_df['OverallQual_GrLivArea'] = train_df['OverallQual_GrLivArea'].clip(upper=80000)  # Increased to 80,000
train_df['GrLivArea_Squared'] = train_df['GrLivArea_Squared'].clip(upper=6000**2)
train_df['TotalSF_Squared'] = train_df['TotalSF_Squared'].clip(upper=12000**2)

# Encode categorical variables
train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)

# Save the scaler
joblib.dump(scaler, f'{BASE_DIR}\\scaler.pkl')
print("Scaler fitted on training data and saved!")

# Step 2: Feature selection with RFE using LinearRegression
print("Training RFE with LinearRegression for feature selection...")
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=50)
rfe.fit(X_train_scaled, y_train)

# Get selected features
selected_features = X_train_scaled.columns[rfe.support_]
print("Selected features:", selected_features)

# Step 3: Train XGBoost with hyperparameter tuning
print("Training XGBoost with hyperparameter tuning...")
xgb_model = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]  # Added for regularization
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)  # Increased to 10-fold CV
grid_search_xgb.fit(X_train_scaled[selected_features], y_train)
xgb_model = grid_search_xgb.best_estimator_
print("Best XGBoost parameters:", grid_search_xgb.best_params_)

# Cross-validation for XGBoost
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled[selected_features], y_train, cv=10, scoring='neg_root_mean_squared_error')
cv_rmse_xgb = -cv_scores_xgb.mean()
cv_std_xgb = cv_scores_xgb.std()
print(f"XGBoost Cross-Validation RMSE (log scale): {cv_rmse_xgb:.4f} (+/- {cv_std_xgb:.4f})")

cv_r2_scores_xgb = cross_val_score(xgb_model, X_train_scaled[selected_features], y_train, cv=10, scoring='r2')
cv_r2_xgb = cv_r2_scores_xgb.mean()
print(f"XGBoost Cross-Validation R²: {cv_r2_xgb:.4f}")

# Step 4: Train LightGBM with hyperparameter tuning
print("Training LightGBM with hyperparameter tuning...")
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)
param_grid_lgbm = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_data_in_leaf': [20, 50]  # Added for regularization
}
grid_search_lgbm = GridSearchCV(lgbm_model, param_grid_lgbm, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)  # Increased to 10-fold CV
grid_search_lgbm.fit(X_train_scaled[selected_features], y_train)
lgbm_model = grid_search_lgbm.best_estimator_
print("Best LightGBM parameters:", grid_search_lgbm.best_params_)

# Cross-validation for LightGBM
cv_scores_lgbm = cross_val_score(lgbm_model, X_train_scaled[selected_features], y_train, cv=10, scoring='neg_root_mean_squared_error')
cv_rmse_lgbm = -cv_scores_lgbm.mean()
cv_std_lgbm = cv_scores_lgbm.std()
print(f"LightGBM Cross-Validation RMSE (log scale): {cv_rmse_lgbm:.4f} (+/- {cv_std_lgbm:.4f})")

cv_r2_scores_lgbm = cross_val_score(lgbm_model, X_train_scaled[selected_features], y_train, cv=10, scoring='r2')
cv_r2_lgbm = cv_r2_scores_lgbm.mean()
print(f"LightGBM Cross-Validation R²: {cv_r2_lgbm:.4f}")

# Step 5: Train CatBoost with hyperparameter tuning
print("Training CatBoost with hyperparameter tuning...")
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
param_grid_catboost = {
    'iterations': [200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [3, 5, 7]  # Added for regularization
}
grid_search_catboost = GridSearchCV(catboost_model, param_grid_catboost, cv=10, scoring='neg_root_mean_squared_error', n_jobs=-1)  # Increased to 10-fold CV
grid_search_catboost.fit(X_train_scaled[selected_features], y_train)
catboost_model = grid_search_catboost.best_estimator_
print("Best CatBoost parameters:", grid_search_catboost.best_params_)

# Cross-validation for CatBoost
cv_scores_catboost = cross_val_score(catboost_model, X_train_scaled[selected_features], y_train, cv=10, scoring='neg_root_mean_squared_error')
cv_rmse_catboost = -cv_scores_catboost.mean()
cv_std_catboost = cv_scores_catboost.std()
print(f"CatBoost Cross-Validation RMSE (log scale): {cv_rmse_catboost:.4f} (+/- {cv_std_catboost:.4f})")

cv_r2_scores_catboost = cross_val_score(catboost_model, X_train_scaled[selected_features], y_train, cv=10, scoring='r2')
cv_r2_catboost = cv_r2_scores_catboost.mean()
print(f"CatBoost Cross-Validation R²: {cv_r2_catboost:.4f}")

# Save the models
joblib.dump(xgb_model, f'{BASE_DIR}\\final_model_xgboost_rfe.pkl')
joblib.dump(lgbm_model, f'{BASE_DIR}\\final_model_lightgbm_rfe.pkl')
joblib.dump(catboost_model, f'{BASE_DIR}\\final_model_catboost_rfe.pkl')
joblib.dump(selected_features, f'{BASE_DIR}\\selected_features.pkl')
print("Models and selected features saved!")

# Step 6: Load and preprocess the Kaggle test set
print("Loading and preprocessing the Kaggle test set...")
test_df = pd.read_csv(f'{BASE_DIR}\\dataset\\test.csv')
test_ids = test_df['Id']
test_df = test_df.drop(columns=['Id'])

# Handle missing values
numerical_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
test_df[numerical_cols] = test_df[numerical_cols].fillna(test_df[numerical_cols].median())
categorical_cols = test_df.select_dtypes(include=['object']).columns
test_df[categorical_cols] = test_df[categorical_cols].fillna(test_df[categorical_cols].mode().iloc[0])

# Feature engineering
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df['HouseAge'] = test_df['YrSold'] - test_df['YearBuilt']
test_df['OverallQual_SF'] = test_df['OverallQual'] * test_df['TotalSF']
test_df['OverallQual_GrLivArea'] = test_df['OverallQual'] * test_df['GrLivArea']
# Add new features
test_df['TotalBathrooms'] = test_df['FullBath'] + 0.5 * test_df['HalfBath'] + test_df['BsmtFullBath'] + 0.5 * test_df['BsmtHalfBath']
test_df['OverallQual_GarageArea'] = test_df['OverallQual'] * test_df['GarageArea']
# Add polynomial features
test_df['GrLivArea_Squared'] = test_df['GrLivArea'] ** 2
test_df['TotalSF_Squared'] = test_df['TotalSF'] ** 2
# Add interaction with Neighborhood
neighborhood_qual_test = pd.get_dummies(test_df['Neighborhood'], prefix='Neighborhood')
interaction_terms_test = pd.DataFrame(index=test_df.index)
for col in neighborhood_qual_test.columns:
    interaction_terms_test[f'{col}_OverallQual'] = neighborhood_qual_test[col] * test_df['OverallQual']
test_df = pd.concat([test_df, interaction_terms_test], axis=1)
# Add new interaction term
test_df['OverallQual_YearBuilt'] = test_df['OverallQual'] * test_df['YearBuilt']
# Add spatial clustering of neighborhoods
neighborhood_encoded_test = pd.get_dummies(test_df['Neighborhood']).values
# Align neighborhood_encoded_test with training set
missing_neighborhoods = set(neighborhood_qual.columns) - set(neighborhood_qual_test.columns)
for col in missing_neighborhoods:
    neighborhood_qual_test[col] = 0
neighborhood_qual_test = neighborhood_qual_test[neighborhood_qual.columns]
neighborhood_encoded_test = neighborhood_qual_test.values
test_df['Neighborhood_Cluster'] = kmeans.predict(neighborhood_encoded_test)

# Cap features in the test set
test_df['GrLivArea'] = test_df['GrLivArea'].clip(upper=6000)
test_df['TotalSF'] = test_df['TotalSF'].clip(upper=12000)
test_df['OverallQual_SF'] = test_df['OverallQual_SF'].clip(upper=80000)
test_df['OverallQual_GrLivArea'] = test_df['OverallQual_GrLivArea'].clip(upper=80000)
test_df['GrLivArea_Squared'] = test_df['GrLivArea_Squared'].clip(upper=6000**2)
test_df['TotalSF_Squared'] = test_df['TotalSF_Squared'].clip(upper=12000**2)

# Encode categorical variables
test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

# Align test set columns with training set columns
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0
extra_cols = set(test_df.columns) - set(train_df.columns)
test_df = test_df.drop(columns=extra_cols)
test_df = test_df[train_df.columns]

# Scale the test set using the fitted scaler
X_test_kaggle_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

print("Kaggle test set preprocessed and scaled successfully!")

# Step 7: Generate the Kaggle Submission File with Stacking
print("Generating predictions for the Kaggle test set with stacking...")
# Subset Kaggle test set to RFE-selected features
X_test_kaggle_rfe = X_test_kaggle_scaled[selected_features]

# Split training data for stacking
X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(X_train_scaled[selected_features], y_train, test_size=0.2, random_state=42)

# Get predictions on validation set for meta-model
y_pred_xgb_meta = xgb_model.predict(X_val_meta)
y_pred_lgbm_meta = lgbm_model.predict(X_val_meta)
y_pred_catboost_meta = catboost_model.predict(X_val_meta)

# Create meta-features
meta_features = np.column_stack((y_pred_xgb_meta, y_pred_lgbm_meta, y_pred_catboost_meta))

# Train meta-model (GradientBoostingRegressor instead of LinearRegression)
meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
meta_model.fit(meta_features, y_val_meta)

# Predict on test set
y_pred_xgb_test = xgb_model.predict(X_test_kaggle_rfe)
y_pred_lgbm_test = lgbm_model.predict(X_test_kaggle_rfe)
y_pred_catboost_test = catboost_model.predict(X_test_kaggle_rfe)
meta_features_test = np.column_stack((y_pred_xgb_test, y_pred_lgbm_test, y_pred_catboost_test))
y_pred_ensemble_log = meta_model.predict(meta_features_test)

# Inspect the range of predicted log(SalePrice) values
print("Min predicted log(SalePrice):", y_pred_ensemble_log.min())
print("Max predicted log(SalePrice):", y_pred_ensemble_log.max())

# Convert predictions back to original SalePrice scale
y_pred_ensemble = np.expm1(y_pred_ensemble_log)

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_ensemble
})

# Save submission file as submission-5.csv
submission.to_csv(f'{BASE_DIR}\\submission-5.csv', index=False)
print("Kaggle submission file saved as 'submission-5.csv'!")

# Verify the submission file
submission_df = pd.read_csv(f'{BASE_DIR}\\submission-5.csv')
print("Submission file shape:", submission_df.shape)
print("Submission file columns:", submission_df.columns)
print("First few rows of submission file:")
print(submission_df.head())
print("Any missing values in submission file:")
print(submission_df.isnull().sum())