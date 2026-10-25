"""
Unit tests for Feature Extraction Engine.
"""

import pytest
from ml.feature_extractor import (
    extract_url_features, normalize_url, calculate_shannon_entropy
)

def test_normalize_url():
    assert normalize_url("google.com") == "http://google.com"
    assert normalize_url("https://secure.bank.com") == "https://secure.bank.com"
    assert normalize_url("   http://test.org/page  ") == "http://test.org/page"

def test_shannon_entropy():
    # Empty string should have 0 entropy
    assert calculate_shannon_entropy("") == 0.0
    # Single character string should have 0 entropy
    assert calculate_shannon_entropy("aaaaa") == 0.0
    # Diverse random string should have high entropy (> 3.0)
    high_entropy = calculate_shannon_entropy("a8f9b2c01d93e87")
    assert high_entropy > 3.0

def test_extract_url_features_benign():
    url = "https://www.wikipedia.org/wiki/Computer_security"
    feats = extract_url_features(url)
    
    assert feats["is_https"] == 1
    assert feats["has_ip_address"] == 0
    assert feats["tld"] == "org"
    assert feats["url_length"] == len(url)
    assert feats["num_slashes"] >= 4
    assert feats["num_underscores"] >= 1
    assert feats["has_suspicious_keyword"] in [0, 1]

def test_extract_url_features_phishing_ip():
    url = "http://192.168.1.50:8080/login/verify.php?token=928a"
    feats = extract_url_features(url)
    
    assert feats["has_ip_address"] == 1
    assert feats["has_port"] == 1
    assert feats["is_https"] == 0
    assert feats["suspicious_keyword_count"] >= 2 # login, verify
    assert feats["has_suspicious_keyword"] == 1

def test_extract_url_features_subdomain_abuse():
    url = "http://paypal.com.account-update.xyz/signin/auth.html"
    feats = extract_url_features(url)
    
    assert feats["num_dots"] >= 3
    assert feats["tld"] == "xyz"
    assert feats["is_suspicious_tld"] == 1
    assert feats["has_suspicious_keyword"] == 1
