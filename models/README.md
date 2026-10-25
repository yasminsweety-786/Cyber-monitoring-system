# Models Directory

This directory stores the trained machine learning models, feature preprocessors, SHAP explainers, and evaluation metrics.

## Artifacts

- `best_model.joblib`: Serialized XGBoost Classifier trained on the 70% training split.
- `preprocessor.joblib`: Serialized `URLPreprocessor` instance fitted exclusively on the training partition (StandardScaler + Top-K TLD OneHotEncoder).
- `explainer.joblib`: Serialized `shap.TreeExplainer` instance for local Shapley value feature attribution.
- `model_metadata.json`: Machine-readable metadata detailing model hyperparameters, data split sizes, feature names, and performance metrics.
- `evaluation_results.json`: Multi-model benchmark results across Logistic Regression, Random Forest, and XGBoost, including confusion matrices and ROC curve coordinates.

## Re-training

To re-train the models and regenerate all artifacts:
```bash
python ml/train.py
```
