"""
Feature Extraction Engine for ML-Based Cyber Monitoring System.
Extracts lexical, structural, statistical, keyword, and domain features from raw URLs.
Ensures zero data leakage and consistent feature generation for both training and real-time inference.
"""

import math
import re
import urllib.parse
from typing import Dict, Any, List, Optional
import tldextract

# High-risk / security-sensitive keywords commonly found in phishing attacks
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "password", "secure", "update", "confirm", 
    "banking", "signin", "auth", "admin", "wallet", "ebayisapi", "webscr", 
    "token", "recover", "service", "portal", "alert", "checkout", "pay",
    "support", "billing", "authenticate", "identity", "credential", "suspend"
]

# TLDs frequently abused in malicious campaigns (used as a numeric signal, NOT a hard rule)
HIGH_ABUSE_TLDS = {
    "xyz", "top", "club", "icu", "work", "buzz", "fit", "gq", "cf", "tk", 
    "ml", "ga", "rest", "cam", "cfd", "monster", "sbs", "quest", "click"
}

# Regex to detect raw IPv4 and IPv6 addresses in hostname
IPV4_REGEX = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
IPV6_REGEX = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")

def calculate_shannon_entropy(text: str) -> float:
    """Calculates the Shannon entropy (randomness) of a given string."""
    if not text:
        return 0.0
    entropy = 0.0
    length = len(text)
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    for count in counts.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    return round(entropy, 4)

def normalize_url(url: str) -> str:
    """Safely trims and adds default scheme if missing for parsing."""
    if not url or not isinstance(url, str):
        return ""
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://", "ftp://")):
        clean_url = "http://" + clean_url
    return clean_url

def extract_url_features(raw_url: str, include_network: bool = False) -> Dict[str, Any]:
    """
    Extracts comprehensive feature dictionary from a URL string.
    Features are strictly computed from the URL text and domain structure.
    
    Parameters:
    - raw_url: The URL string to inspect.
    - include_network: If True, optionally perform safe HEAD probe for redirects.
    
    Returns:
    - Dict of numeric and categorical features.
    """
    url = normalize_url(raw_url)
    
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        # Fallback for malformed strings
        parsed = urllib.parse.urlparse("http://malformed-url.local")

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment
    
    # Remove port from netloc for domain analysis
    hostname = netloc.split(":")[0] if ":" in netloc else netloc
    has_port = 1 if ":" in netloc else 0

    # 1. Structural & Length Features
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # Character counts
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscores = url.count("_")
    num_slashes = url.count("/")
    num_questionmarks = url.count("?")
    num_equal = url.count("=")
    num_at = url.count("@")
    num_ampersand = url.count("&")
    num_percent = url.count("%")
    num_tilde = url.count("~")
    num_plus = url.count("+")
    num_exclamation = url.count("!")
    num_dollar = url.count("$")
    num_hashtag = url.count("#")

    # Digit counts & ratios
    num_digits = sum(c.isdigit() for c in url)
    digit_ratio = round(num_digits / max(url_length, 1), 4)

    # Path & Query Metrics
    path_segments = [seg for seg in path.split("/") if seg]
    num_subdirectories = len(path_segments)
    
    query_params = urllib.parse.parse_qs(query, keep_blank_values=True)
    num_params = len(query_params)
    
    has_double_slash_in_path = 1 if "//" in path else 0

    # 2. IP Address Detection
    has_ip = 1 if (IPV4_REGEX.match(hostname) or (IPV6_REGEX.match(hostname) and ":" in hostname)) else 0

    # 3. Protocol & HTTPS Features
    is_https = 1 if scheme == "https" else 0
    has_http_in_path_or_host = 1 if ("http" in path.lower() or "https" in path.lower() or "http" in hostname[4:]) else 0

    # 4. Domain & TLD Features using tldextract
    ext = tldextract.extract(hostname)
    # Extract domain components cleanly
    subdomain = ext.subdomain or ""
    suffix = (ext.suffix or "").lower()
    domain_name = ext.domain or ""
    if domain_name and suffix:
        registered_domain = f"{domain_name}.{suffix}"
    elif domain_name:
        registered_domain = domain_name
    else:
        registered_domain = hostname

    domain_length = len(registered_domain) if registered_domain else len(hostname)
    num_subdomains = len([s for s in subdomain.split(".") if s]) if subdomain else 0
    is_punycode = 1 if "xn--" in hostname else 0
    is_suspicious_tld = 1 if suffix in HIGH_ABUSE_TLDS else 0
    tld_length = len(suffix)

    # 5. Keyword & Lexical Indicators
    url_lower = url.lower()
    keyword_matches = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
    suspicious_keyword_count = len(keyword_matches)
    has_suspicious_keyword = 1 if suspicious_keyword_count > 0 else 0

    # Check for brand spoofing / multiple domain names inside path or query
    has_client_server_in_domain = 1 if ("server" in hostname or "client" in hostname or "admin" in hostname) else 0

    # 6. Statistical & Entropy Features
    domain_entropy = calculate_shannon_entropy(registered_domain if registered_domain else hostname)
    url_entropy = calculate_shannon_entropy(url)

    # 7. Safe Network Probe (Only when requested, default 0 for training)
    redirect_count = 0
    if include_network:
        try:
            import requests
            resp = requests.head(url, timeout=2.0, allow_redirects=True, headers={'User-Agent': 'CyberScanner-Bot/1.0'})
            redirect_count = len(resp.history)
        except Exception:
            redirect_count = 0

    features = {
        # Length & Counts
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "domain_length": domain_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_questionmarks": num_questionmarks,
        "num_equal": num_equal,
        "num_at": num_at,
        "num_ampersand": num_ampersand,
        "num_percent": num_percent,
        "num_tilde": num_tilde,
        "num_plus": num_plus,
        "num_exclamation": num_exclamation,
        "num_dollar": num_dollar,
        "num_hashtag": num_hashtag,
        "num_digits": num_digits,
        "digit_ratio": digit_ratio,
        "num_subdirectories": num_subdirectories,
        "num_params": num_params,
        "num_subdomains": num_subdomains,
        
        # Binary structural flags
        "has_ip_address": has_ip,
        "has_port": has_port,
        "has_double_slash_in_path": has_double_slash_in_path,
        "is_https": is_https,
        "has_http_in_path_or_host": has_http_in_path_or_host,
        "is_punycode": is_punycode,
        "is_suspicious_tld": is_suspicious_tld,
        "has_client_server_in_domain": has_client_server_in_domain,
        
        # Keywords
        "suspicious_keyword_count": suspicious_keyword_count,
        "has_suspicious_keyword": has_suspicious_keyword,
        
        # Entropy & TLD
        "tld_length": tld_length,
        "domain_entropy": domain_entropy,
        "url_entropy": url_entropy,
        "redirect_count": redirect_count,
        "tld": suffix if suffix else "none"
    }

    return features

def extract_features_dataframe(urls: List[str]) -> Any:
    """Extracts features for a batch of URLs into a pandas DataFrame."""
    import pandas as pd
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=cybersecurity",
        "http://paypal-security-update.com.account-verify.xyz/login.php?token=928a&user=admin",
        "http://192.168.1.1:8080/admin/index.html",
        "https://stanford.edu/admissions/apply"
    ]
    for u in test_urls:
        print(f"\nURL: {u}")
        feat = extract_url_features(u)
        print(f"Features preview: length={feat['url_length']}, has_ip={feat['has_ip_address']}, keywords={feat['suspicious_keyword_count']}, entropy={feat['domain_entropy']}, tld={feat['tld']}")
