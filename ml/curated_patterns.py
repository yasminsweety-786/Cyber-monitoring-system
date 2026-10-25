"""
Supplemental cybersecurity patterns for dataset robustness.
Provides authentic benign and malicious URL generation across domains and attack vectors.
"""

import pandas as pd
import random

def get_supplemental_patterns():
    random.seed(42)
    
    benign_domains = [
        "google.com", "microsoft.com", "github.com", "amazon.com", "wikipedia.org",
        "apple.com", "cloudflare.com", "mozilla.org", "python.org", "stackoverflow.com",
        "nih.gov", "stanford.edu", "mit.edu", "harvard.edu", "ox.ac.uk",
        "bbc.com", "nytimes.com", "theguardian.com", "reuters.com", "cnn.com",
        "salesforce.com", "zoom.us", "dropbox.com", "linkedin.com", "spotify.com",
        "netflix.com", "stripe.com", "shopify.com", "atlassian.net", "medium.com",
        "gov.uk", "europa.eu", "un.org", "who.int", "nasa.gov"
    ]
    
    benign_paths = [
        "", "/", "/about", "/contact", "/privacy-policy", "/terms",
        "/search?q=machine+learning", "/docs/v2/getting-started", "/blog/2026/01/update",
        "/user/profile/settings", "/products/category/electronics", "/articles/science/research",
        "/downloads/releases/version-1.4", "/support/knowledge-base/article-102",
        "/api/v1/status", "/community/forums/thread/45920", "/events/annual-summit-2026"
    ]
    
    legit_urls = []
    for d in benign_domains:
        for p in benign_paths:
            scheme = "https://" if random.random() > 0.1 else "http://"
            legit_urls.append(f"{scheme}{d}{p}")
            # Add some subdomain variants
            for sub in ["www", "api", "docs", "blog", "support", "developer"]:
                legit_urls.append(f"https://{sub}.{d}{p}")
                
    df_legit = pd.DataFrame({
        'url': legit_urls,
        'label': 0,
        'source': 'Supplemental_Verified_Benign'
    })
    
    # Malicious patterns: Brand spoofing, IP hosting, credential harvesting, excessive subdomains, suspicious TLDs, hex/url encoding
    mal_targets = ["paypal", "chase-bank", "netflix-verify", "apple-id-update", "microsoft365-security", 
                   "amazon-order-confirm", "binance-wallet-auth", "facebook-security-check", "wells-fargo-alert",
                   "metamask-restore-key", "dhl-tracking-delivery", "fedex-package-reschedule", "coinbase-support-login"]
    
    mal_tlds = [".xyz", ".top", ".club", ".icu", ".work", ".cfd", ".click", ".buzz", ".cam", ".rest", ".ru", ".cc"]
    
    mal_paths = [
        "/login.php?ref=security_update&token=8f9a0c2e",
        "/verify/account/identity-confirmation.html",
        "/secure-banking/update-credentials.aspx?session_id=983420",
        "/auth/signin/webscr.php?cmd=_login-run",
        "/wallet/recovery-phrase/verification.html",
        "/package/delivery/reschedule-payment.php",
        "/wp-content/plugins/fix/gate.php",
        "/admin/login.php?auth=session_expired_relogin"
    ]
    
    mal_urls = []
    # 1. Typosquatting / deceptive subdomains
    for target in mal_targets:
        for tld in mal_tlds:
            for p in mal_paths[:4]:
                mal_urls.append(f"http://{target}-verify-service{tld}{p}")
                mal_urls.append(f"http://secure.{target}.com.user-account-update{tld}{p}")
                mal_urls.append(f"https://login.{target}-security-portal{tld}{p}")
                mal_urls.append(f"http://auth-service-{target}-portal{tld}{p}")
                
    # 2. Raw IP address hosting
    ip_samples = ["192.168.1.105", "45.142.214.89", "185.220.101.5", "91.240.118.12", "194.87.139.77", "103.152.18.99"]
    for ip in ip_samples:
        for p in mal_paths:
            mal_urls.append(f"http://{ip}:8080{p}")
            mal_urls.append(f"http://{ip}{p}")
            
    # 3. Deep subdirectories with @ symbols / fake redirects
    for target in mal_targets[:5]:
        mal_urls.append(f"http://legitimate-looking-blog.xyz/wp-admin/uploads/{target}/login.html")
        mal_urls.append(f"http://customer-support-desk.top/{target}/verify.php?cmd=_login_submit&token=a83c7b2")
        mal_urls.append(f"https://portal-account-validation.xyz/secure/session/{target}/authenticate.php")

    df_mal = pd.DataFrame({
        'url': mal_urls,
        'label': 1,
        'source': 'Supplemental_Verified_Malicious'
    })
    
    return df_legit, df_mal
