"""
Unit tests for URLPreprocessor transformation pipeline.
"""

import pytest
import numpy as np
from ml.preprocessor import URLPreprocessor

def test_preprocessor_fitting_and_transform():
    urls = [
        "https://google.com/search?q=test",
        "http://phish-login.xyz/verify",
        "https://stanford.edu/admissions",
        "http://192.168.1.1/admin"
    ]
    prep = URLPreprocessor()
    X = prep.fit_transform(urls)
    
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(urls)
    assert X.shape[1] == 60 # total feature count
    assert prep.is_fitted is True

def test_preprocessor_unseen_tld():
    prep = URLPreprocessor()
    prep.fit(["https://google.com"])
    
    # Test with exotic TLD that maps to 'other'
    X_new = prep.transform("https://some-rare-domain.unknowncustomtld/path")
    assert X_new.shape == (1, 60)
    assert not np.isnan(X_new).any()
