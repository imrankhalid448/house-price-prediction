3. The script will generate:
- `submission-5.csv`: Latest Kaggle submission file (score 0.13257).
- `final_model_xgboost_rfe.pkl`: Trained XGBoost model.
- `final_model_lightgbm_rfe.pkl`: Trained LightGBM model.
- `final_model_catboost_rfe.pkl`: Trained CatBoost model.
- `selected_features.pkl`: RFE-selected features.
- `scaler.pkl`: Fitted scaler.
4. Note: To reproduce the best submission (score 0.12565), refer to the script version before adding CatBoost and the GradientBoosting meta-model (not included in this repository).

## Results
- **Validation RMSE (Original Linear Regression):** 0.1226 (log scale).
- **Validation R² (Original Linear Regression):** 0.9185.
- **Cross-Validation RMSE (Linear Regression):** 0.1449 (log scale).
- **Cross-Validation R² (Linear Regression):** 0.8645.
- **Kaggle Score (Initial Submission, Linear Regression):** 0.18241 (log scale), top 40-50%.
- **Cross-Validation RMSE (XGBoost, untuned):** 0.1325 (log scale).
- **Cross-Validation R² (XGBoost, untuned):** 0.8894.
- **Kaggle Score (Intermediate Submission, XGBoost):** 0.13195 (log scale), top 15-20%.
- **Cross-Validation RMSE (XGBoost, tuned, best submission):** 0.1216 (log scale).
- **Cross-Validation R² (XGBoost, tuned, best submission):** 0.9069.
- **Cross-Validation RMSE (LightGBM, tuned, best submission):** 0.1253 (log scale).
- **Cross-Validation R² (LightGBM, tuned, best submission):** 0.9009.
- **Kaggle Score (Best Submission, XGBoost + LightGBM Stacking):** 0.12565 (log scale), rank 659, top 10-15%.
- **Cross-Validation RMSE (XGBoost, tuned, final submission):** 0.1198 (log scale).
- **Cross-Validation R² (XGBoost, tuned, final submission):** 0.9082.
- **Cross-Validation RMSE (LightGBM, tuned, final submission):** 0.1237 (log scale).
- **Cross-Validation R² (LightGBM, tuned, final submission):** 0.9023.
- **Cross-Validation RMSE (CatBoost, tuned, final submission):** 0.1176 (log scale).
- **Cross-Validation R² (CatBoost, tuned, final submission):** 0.9121.
- **Kaggle Score (Intermediate Submission, XGBoost + LightGBM + CatBoost Stacking):** 0.12728 (log scale), top 10-15%.
- **Kaggle Score (Final Submission, XGBoost + LightGBM + CatBoost Stacking with GradientBoosting Meta-Model):** 0.13257 (log scale), top 15-20%.

## Submission to Kaggle
1. The best submission file is `submission.csv` in the main project directory (score 0.12565, rank 659).
2. Submit it to the Kaggle competition: [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Additional Information
- See `report/project_report.md` for a detailed project report, including methodology, challenges, and conclusion.
- Visualizations are available in the `plots/` directory.