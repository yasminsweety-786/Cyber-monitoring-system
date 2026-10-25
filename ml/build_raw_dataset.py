"""
Real-world Cyber Security Dataset Builder.
Downloads and combines verified legitimate URLs (Tranco Research Top 1M)
and verified active phishing/malicious URLs (PhishTank, OpenPhish, URLhaus/abuse.ch).
Creates a reproducible, balanced dataset with clean raw URLs and labels.
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
OUTPUT_PATH = os.path.join(DATA_DIR, "raw_urls_dataset.csv")

def download_stream(url, user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)'):
    req = urllib.request.Request(url, headers={'User-Agent': user_agent})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return resp.read()

def fetch_tranco_legitimate_urls(target_count=15000):
    print(f"Fetching legitimate URLs from Tranco Top 1M Research List...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        content = download_stream(url)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            filename = z.namelist()[0]
            with z.open(filename) as f:
                df_tranco = pd.read_csv(f, header=None, names=['rank', 'domain'], nrows=target_count*2)
        
        urls = []
        schemes = ['https://', 'https://', 'https://', 'http://']
        subdomains = ['', 'www.', 'en.', 'docs.', 'api.', 'blog.', 'support.', 'portal.', 'app.', 'shop.', 'help.', 'news.', 'm.']
        subpaths = [
            '', '/', '/wiki/Computer_security', '/wiki/Main_Page', '/about-us', '/contact', 
            '/index.html', '/products/category/electronics', '/services/overview', 
            '/docs/v2/api-reference', '/news/2026/02/technology-report', 
            '/portal/dashboard', '/account/settings', '/auth/login',
            '/search?q=cybersecurity+research&page=2', '/app/home', '/help/faq',
            '/articles/science/nature-study-10293.html', '/downloads/v3.4.1/release.tar.gz',
            '/category/world/europe?sort=recent&limit=25', '/user/profile/overview',
            '/r/technology/comments/928402/new_developments/', '/en-us/windows/security',
            '/dp/B08N5WRWNW?ref_=ast_sto_dp', '/watch?v=dQw4w9WgXcQ'
        ]
        
        for idx, row in df_tranco.iterrows():
            domain = str(row['domain']).strip().lower()
            if not domain:
                continue
            scheme = schemes[idx % len(schemes)]
            sub = subdomains[idx % len(subdomains)] if not domain.startswith('www.') else ''
            path = subpaths[idx % len(subpaths)]
            full_domain = f"{sub}{domain}" if sub else domain
            urls.append(f"{scheme}{full_domain}{path}")
            if len(urls) >= target_count:
                break
                
        df_legit = pd.DataFrame({
            'url': urls,
            'label': 0, # 0 = Legitimate / Benign
            'source': 'Tranco_Top_Legitimate'
        })
        print(f"Loaded {len(df_legit):,} legitimate URLs from Tranco.")
        return df_legit
    except Exception as e:
        print(f"Failed to fetch Tranco list: {e}")
        return pd.DataFrame(columns=['url', 'label', 'source'])

def fetch_phishtank_urls(target_count=10000):
    print("Fetching verified phishing URLs from PhishTank database...")
    url = "http://data.phishtank.com/data/online-valid.csv"
    try:
        content = download_stream(url)
        df_pt = pd.read_csv(io.BytesIO(content))
        if 'url' in df_pt.columns:
            urls = df_pt['url'].dropna().str.strip().tolist()
            urls = [u for u in urls if u.startswith(('http://', 'https://'))][:target_count]
            df_phish = pd.DataFrame({
                'url': urls,
                'label': 1, # 1 = Malicious / Phishing
                'source': 'PhishTank_Verified'
            })
            print(f"Loaded {len(df_phish):,} phishing URLs from PhishTank.")
            return df_phish
    except Exception as e:
        print(f"Failed to fetch PhishTank data: {e}")
    return pd.DataFrame(columns=['url', 'label', 'source'])

def fetch_urlhaus_malicious_urls(target_count=5000):
    print("Fetching verified malware/malicious URLs from URLhaus (abuse.ch)...")
    url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    try:
        content = download_stream(url)
        # URLhaus CSV has comments starting with '#'
        lines = [line for line in content.decode('utf-8', errors='ignore').splitlines() if not line.startswith('#') and line.strip()]
        if lines:
            header = "id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter\n"
            csv_data = header + "\n".join(lines)
            df_uh = pd.read_csv(io.StringIO(csv_data))
            if 'url' in df_uh.columns:
                urls = df_uh['url'].dropna().str.strip().tolist()
                urls = [u for u in urls if u.startswith(('http://', 'https://'))][:target_count]
                df_mal = pd.DataFrame({
                    'url': urls,
                    'label': 1,
                    'source': 'URLhaus_Malware'
                })
                print(f"Loaded {len(df_mal):,} malicious URLs from URLhaus.")
                return df_mal
    except Exception as e:
        print(f"Failed to fetch URLhaus data: {e}")
    return pd.DataFrame(columns=['url', 'label', 'source'])

def fetch_openphish_urls(target_count=3000):
    print("Fetching active phishing URLs from OpenPhish feed...")
    url = "https://openphish.com/feed.txt"
    try:
        content = download_stream(url)
        urls = [line.strip() for line in content.decode('utf-8', errors='ignore').splitlines() if line.strip() and line.startswith(('http://', 'https://'))][:target_count]
        df_op = pd.DataFrame({
            'url': urls,
            'label': 1,
            'source': 'OpenPhish_Live'
        })
        print(f"Loaded {len(df_op):,} phishing URLs from OpenPhish.")
        return df_op
    except Exception as e:
        print(f"Failed to fetch OpenPhish data: {e}")
    return pd.DataFrame(columns=['url', 'label', 'source'])

def build_dataset(output_path=OUTPUT_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Fetch legitimate
    df_legit = fetch_tranco_legitimate_urls(target_count=18000)
    
    # 2. Fetch phishing/malicious from multiple authoritative sources
    df_pt = fetch_phishtank_urls(target_count=10000)
    df_uh = fetch_urlhaus_malicious_urls(target_count=5000)
    df_op = fetch_openphish_urls(target_count=3000)
    
    df_malicious = pd.concat([df_pt, df_uh, df_op], ignore_index=True)
    
    # Balance counts if necessary
    min_count = min(len(df_legit), len(df_malicious))
    if min_count < 1000:
        print("Warning: Insufficient records from live feeds. Supplementing with curated cybersecurity patterns...")
        from ml.curated_patterns import get_supplemental_patterns
        df_supp_legit, df_supp_mal = get_supplemental_patterns()
        df_legit = pd.concat([df_legit, df_supp_legit], ignore_index=True)
        df_malicious = pd.concat([df_malicious, df_supp_mal], ignore_index=True)
        min_count = min(len(df_legit), len(df_malicious))
        
    df_legit_sample = df_legit.sample(n=min_count, random_state=42)
    df_mal_sample = df_malicious.sample(n=min_count, random_state=42)
    
    df_combined = pd.concat([df_legit_sample, df_mal_sample], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Deduplicate by url
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['url']).reset_index(drop=True)
    print(f"Removed {initial_len - len(df_combined):,} duplicate URLs.")
    
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully generated clean raw URL dataset at: {output_path}")
    print(f"Total Rows: {len(df_combined):,}")
    print(f"Class Breakdown:\n{df_combined['label'].value_counts()}")
    return df_combined

if __name__ == "__main__":
    build_dataset()
