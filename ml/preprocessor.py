import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from ml.feature_extractor import extract_url_features, extract_features_dataframe

# List of top TLDs to explicitly encode; any other TLD gets categorized as 'other'
TOP_TLDS = [
    "com", "org", "net", "edu", "gov", "io", "co", "xyz", "top", 
    "ru", "cn", "in", "uk", "de", "app", "info", "biz", "online", 
    "site", "club", "icu", "none"
]

NUMERICAL_FEATURES = [
    "url_length", "hostname_length", "path_length", "query_length", "domain_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_questionmarks",
    "num_equal", "num_at", "num_ampersand", "num_percent", "num_tilde", "num_plus",
    "num_exclamation", "num_dollar", "num_hashtag", "num_digits", "digit_ratio",
    "num_subdirectories", "num_params", "num_subdomains", "tld_length",
    "domain_entropy", "url_entropy"
]

BINARY_FEATURES = [
    "has_ip_address", "has_port", "has_double_slash_in_path", "is_https",
    "has_http_in_path_or_host", "is_punycode", "is_suspicious_tld",
    "has_client_server_in_domain", "has_suspicious_keyword", "suspicious_keyword_count"
]

CATEGORICAL_FEATURES = ["tld"]

class URLPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust end-to-end preprocessor for URL features.
    Transforms raw URL features into standardized numeric matrices for ML models.
    """
    def __init__(self, top_tlds: Optional[List[str]] = None):
        self.top_tlds = top_tlds or TOP_TLDS
        self.numerical_cols = NUMERICAL_FEATURES
        self.binary_cols = BINARY_FEATURES
        self.categorical_cols = CATEGORICAL_FEATURES
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            categories=[self.top_tlds + ["other"]], 
            handle_unknown="ignore", 
            sparse_output=False
        )
        self.feature_names_out: List[str] = []
        self.is_fitted = False

    def _prepare_df(self, X: Any) -> pd.DataFrame:
        """Accepts either a list of raw URLs, a single URL string, or a DataFrame of extracted features."""
        if isinstance(X, str):
            feats = [extract_url_features(X)]
            df = pd.DataFrame(feats)
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
            feats = [extract_url_features(u) for u in X]
            df = pd.DataFrame(feats)
        elif isinstance(X, pd.DataFrame):
            # Check if it has 'url' column or extracted features
            if 'url' in X.columns and 'url_length' not in X.columns:
                feats = [extract_url_features(u) for u in X['url']]
                df = pd.DataFrame(feats)
            else:
                df = X.copy()
        else:
            raise ValueError(f"Unsupported input type to URLPreprocessor: {type(X)}")

        # Ensure all expected columns exist with defaults
        for col in self.numerical_cols + self.binary_cols:
            if col not in df.columns:
                df[col] = 0
        if 'tld' not in df.columns:
            df['tld'] = 'none'

        # Map rare TLDs to 'other'
        df['tld'] = df['tld'].apply(lambda t: t if t in self.top_tlds else 'other')
        return df

    def fit(self, X: Any, y: Any = None):
        df = self._prepare_df(X)
        
        # Fit scaler on numerical features
        self.scaler.fit(df[self.numerical_cols])
        
        # Fit one-hot encoder on categorical features
        self.encoder.fit(df[self.categorical_cols])
        
        # Build feature names list
        cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names_out = self.numerical_cols + self.binary_cols + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("URLPreprocessor must be fitted before transforming data.")
        df = self._prepare_df(X)
        
        num_scaled = self.scaler.transform(df[self.numerical_cols])
        bin_vals = df[self.binary_cols].values.astype(float)
        cat_encoded = self.encoder.transform(df[self.categorical_cols])
        
        X_processed = np.hstack([num_scaled, bin_vals, cat_encoded])
        return X_processed

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_out

    def transform_to_df(self, X: Any) -> pd.DataFrame:
        arr = self.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_out)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved URLPreprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "URLPreprocessor":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        return joblib.load(filepath)

if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://paypal-security-update.com.account-verify.xyz/login.php?token=928a&user=admin",
        "http://192.168.1.1:8080/admin/index.html"
    ]
    prep = URLPreprocessor()
    transformed = prep.fit_transform(test_urls)
    print(f"Transformed matrix shape: {transformed.shape}")
    print(f"Feature names count: {len(prep.get_feature_names_out())}")
    print(f"Feature names sample: {prep.get_feature_names_out()[:10]} ... {prep.get_feature_names_out()[-5:]}")
