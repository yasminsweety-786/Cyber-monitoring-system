"""
Statistical Matrices & Visualization Generator.
Generates publication-quality charts for Google Colab and Academic Reports:
1. Correlation Matrix Heatmap (Features vs Malicious Label)
2. Confusion Matrix Heatmap with Exact Counts and Rates
3. Multi-Model ROC Curves (Logistic Regression vs Random Forest vs XGBoost)
4. Feature Importance Bar Plot
5. SHAP Beeswarm Feature Impact Plot
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "visualizations")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def generate_all_matrices():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # 1. Load Data & Models
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Training or test split not found. Running train.py...")
        from ml.train import run_pipeline
        run_pipeline()
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    from ml.feature_extractor import extract_url_features
    from ml.preprocessor import URLPreprocessor
    
    preprocessor = URLPreprocessor.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "r") as f:
        eval_results = json.load(f)

    # -------------------------------------------------------------
    # 1. CORRELATION MATRIX HEATMAP
    # -------------------------------------------------------------
    print("Generating Correlation Matrix Heatmap...")
    # Extract features for a representative sample
    sample_df = train_df.sample(n=min(2000, len(train_df)), random_state=42)
    feat_rows = [extract_url_features(u) for u in sample_df['url']]
    feat_df = pd.DataFrame(feat_rows)
    feat_df['is_malicious'] = sample_df['label'].values
    
    # Select key numerical and binary features for clear heatmap
    key_features = [
        'url_length', 'domain_length', 'path_length', 'num_dots', 'num_subdomains',
        'num_digits', 'digit_ratio', 'num_slashes', 'num_params', 'domain_entropy',
        'url_entropy', 'suspicious_keyword_count', 'has_ip_address', 'is_https',
        'is_suspicious_tld', 'is_malicious'
    ]
    available_cols = [c for c in key_features if c in feat_df.columns]
    corr = feat_df[available_cols].corr()

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
    )
    plt.title("Feature Correlation Matrix Heatmap (URL Cybersecurity Indicators)", fontsize=15, pad=15, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    # -------------------------------------------------------------
    # 2. CONFUSION MATRIX HEATMAP
    # -------------------------------------------------------------
    print("Generating Confusion Matrix Heatmap...")
    xgb_metrics = eval_results["models"]["XGBoost"]["test"]
    cm_data = xgb_metrics["confusion_matrix"]
    cm_arr = np.array([
        [cm_data["tn"], cm_data["fp"]],
        [cm_data["fn"], cm_data["tp"]]
    ])
    
    total_test = cm_arr.sum()
    cm_labels = np.array([
        [f"True Negative (TN)\n{cm_data['tn']:,}\n({cm_data['tn']/total_test*100:.1f}%)", 
         f"False Positive (FP)\n{cm_data['fp']:,}\n({cm_data['fp']/total_test*100:.1f}%)"],
        [f"False Negative (FN)\n{cm_data['fn']:,}\n({cm_data['fn']/total_test*100:.1f}%)", 
         f"True Positive (TP)\n{cm_data['tp']:,}\n({cm_data['tp']/total_test*100:.1f}%)"]
    ])

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_arr, 
        annot=cm_labels, 
        fmt="", 
        cmap="Blues", 
        cbar=True,
        xticklabels=["Predicted Legitimate (0)", "Predicted Malicious (1)"],
        yticklabels=["Actual Legitimate (0)", "Actual Malicious (1)"],
        linewidths=1.5,
        linecolor='white'
    )
    plt.title(f"XGBoost Confusion Matrix (Held-out Test Set: {total_test:,} Samples)\nAccuracy: {xgb_metrics['accuracy']*100:.2f}% | F1-Score: {xgb_metrics['f1_score']:.4f}", fontsize=13, pad=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # -------------------------------------------------------------
    # 3. ROC CURVES MULTI-MODEL COMPARISON
    # -------------------------------------------------------------
    print("Generating ROC Curve Comparison...")
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance Baseline (AUC = 0.50)')
    
    colors = {'Logistic Regression': '#6b7280', 'Random Forest': '#3b82f6', 'XGBoost': '#10b981'}
    for name, m_info in eval_results["models"].items():
        roc_pts = m_info["test"]["roc_curve"]
        auc_val = m_info["test"]["roc_auc"]
        fpr_list = [pt["fpr"] for pt in roc_pts]
        tpr_list = [pt["tpr"] for pt in roc_pts]
        plt.plot(fpr_list, tpr_list, lw=2.5, color=colors.get(name, '#f59e0b'), label=f"{name} (AUC = {auc_val:.4f})")

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=8)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12, labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC) Comparison on Test Split', fontsize=14, pad=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")

    # -------------------------------------------------------------
    # 4. TOP FEATURE IMPORTANCE BAR CHART
    # -------------------------------------------------------------
    print("Generating Feature Importance Bar Chart...")
    top_feats = eval_results.get("top_features", [])[:15]
    if top_feats:
        feat_names = [f["feature"] for f in top_feats][::-1]
        feat_scores = [f["importance"] for f in top_feats][::-1]

        plt.figure(figsize=(10, 7.5))
        bars = plt.barh(feat_names, feat_scores, color='#06b6d4', edgecolor='#0891b2', height=0.65)
        plt.xlabel('Relative Feature Importance (Information Gain / Gini)', fontsize=12, labelpad=8)
        plt.title('Top 15 Most Discriminative Features in URL Threat Classification', fontsize=14, pad=12, fontweight='bold')
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va='center', fontsize=9.5, fontweight='bold')
        plt.xlim(0, max(feat_scores) * 1.15)
        plt.tight_layout()
        fi_path = os.path.join(OUTPUT_DIR, "feature_importance_plot.png")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"Saved: {fi_path}")

    # -------------------------------------------------------------
    # 5. SHAP BEESWARM SUMMARY PLOT
    # -------------------------------------------------------------
    print("Generating SHAP Summary Plot...")
    try:
        import shap
        explainer_path = os.path.join(MODELS_DIR, "explainer.joblib")
        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
            # Sample 250 test points for SHAP summary
            sample_test_urls = test_df['url'].sample(n=min(250, len(test_df)), random_state=42).tolist()
            X_sample = preprocessor.transform(sample_test_urls)
            feature_names = preprocessor.get_feature_names_out()
            
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            elif len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]
                
            plt.figure(figsize=(11, 8))
            shap.summary_plot(
                shap_vals, 
                X_sample, 
                feature_names=feature_names, 
                max_display=15, 
                show=False
            )
            plt.title("SHAP Feature Impact on Model Output (Beeswarm Summary)", fontsize=14, pad=15, fontweight='bold')
            plt.tight_layout()
            shap_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {shap_path}")
    except Exception as e:
        print(f"SHAP plot notice: {e}")

    print("\nAll statistical matrices and visualizations generated successfully in docs/visualizations/!")

if __name__ == "__main__":
    generate_all_matrices()
