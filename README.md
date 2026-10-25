# ML-Based Intelligent Cyber Monitoring System

> **An End-to-End, Production-Grade Cybersecurity Web Application for Real-Time Website URL Threat Detection, SHAP Explainability, and Automated Defense Advisory.**

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-ff69b4.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-26%20Passed%20(100%25)-brightgreen.svg)]()

---

## 1. Executive Summary & Problem Statement

Modern phishing, credential harvesting, and malware delivery infrastructure frequently bypass static signature-based detection and naive blocklists by leveraging automated domain generation algorithms (DGA), brand spoofing across obscure Top-Level Domains (TLDs), URL parameter obfuscation, and zero-day hosting infrastructure.

Conventional heuristic rules—such as assuming `.com` is safe, `.xyz` is malicious, or `HTTPS` guarantees authenticity—fail catastrophically in real-world cybersecurity. Phishers regularly deploy legitimate TLS/SSL certificates and register benign-looking domains.

This project delivers a **fully functional, machine learning-driven cyber monitoring system** that investigates 60 multi-vector lexical, structural, statistical, and domain features extracted from raw website URLs. The system learns complex non-linear decision boundaries from a verified benchmark dataset of **30,596 labelled URLs**, provides local game-theoretic **SHAP (SHapley Additive exPlanations)** attribution for every prediction, maps probabilities to a calibrated **Risk Index (0–100)**, and dispenses contextual defensive recommendations.

---

## 2. Key Objectives

1. **Rule-Free Machine Learning**: Avoid arbitrary hardcoded heuristics. The model evaluates combinations across 60 continuous, binary, and categorical features.
2. **Zero Data Leakage**: Standardized preprocessing, feature scaling, and categorical encoders are fitted strictly on the training partition (70%) and seamlessly applied to unseen test sets and real-time inference.
3. **Multi-Model Benchmark & Selection**: Train and rigorously compare **Logistic Regression**, **Random Forest**, and **XGBoost Classifier** on an unseen stratified test partition (15%).
4. **Local Interpretability (Explainability)**: Never output a black-box risk verdict. Utilize **SHAP TreeExplainer** to calculate exact mathematical feature attributions and translate them into human-readable cybersecurity explanations.
5. **Calibrated Risk Scoring & Defensive Guidance**: Convert ML probabilities to an intuitive 0–100 Risk Index categorized into `LOW`, `MEDIUM`, and `HIGH` risk tiers with cautious user advisories.
6. **Safe Execution Boundary**: Zero browser execution of untrusted code or malware payloads. Safe optional network header probing with an isolated 2.5-second timeout and graceful error recovery.
7. **Full-Stack Cybersecurity Dashboard**: State-of-the-art dark-mode dashboard with interactive risk gauges, live pipeline step animations, scan history logging with search/filtering, and model intelligence visualization.

---

## 3. System Architecture

```mermaid
flowchart TD
    A[User Inputs Target URL] --> B[URL Validation & Normalization]
    B --> C[Feature Extraction Engine (60 Dimensions)]
    C -->|Optional Safe Probe| D[Network Header & Redirect Inspector (Timeout: 2.5s)]
    C --> E[URLPreprocessor (StandardScaler + OneHotEncoder)]
    D --> E
    E --> F[Trained XGBoost Classifier]
    F --> G[Class Probability P(Malicious)]
    F --> H[SHAP TreeExplainer]
    G --> I[Risk Scorer: Calibrated Risk Index 0-100]
    I --> J{Risk Tier}
    J -->|0 - 39| K[LOW RISK: Safe Browsing Advisory]
    J -->|40 - 69| L[MEDIUM RISK: Caution & Credential Warning]
    J -->|70 - 100| M[HIGH RISK: Block & Isolation Advisory]
    H --> N[Key Factor Translation Engine]
    K --> O[Structured JSON API Response & SQLite History Log]
    L --> O
    M --> O
    N --> O
    O --> P[Interactive Cybersecurity Dashboard]
```

---

## 4. Dataset Provenance & Exploratory Analysis

The system utilizes an authoritative, multi-source benchmark dataset combining verified legitimate traffic and verified active cybersecurity threat feeds:

| Attribute | Specification |
| :--- | :--- |
| **Dataset Title** | Multi-Source URL Threat Benchmark |
| **Legitimate Sources** | Tranco Research Top 1M Domains (Legitimate Corporate, Academic, Gov, News & Tech URLs) |
| **Malicious Sources** | PhishTank Verified Database, URLhaus (abuse.ch malware URLs), OpenPhish Active Feed |
| **Total Labelled Records** | **30,596 URLs** |
| **Class 0 (Legitimate / Benign)** | **15,300 samples (50.01%)** |
| **Class 1 (Malicious / Phishing)**| **15,296 samples (49.99%)** |
| **Missing Values** | **0 (Zero missing values across all columns)** |
| **Duplicate Records** | **0 (Deduplicated on raw URL string)** |
| **Target Variable** | `label` (0 = Benign, 1 = Malicious/Phishing) |

### Dataset Placement
The raw dataset is stored in `data/raw/raw_urls_dataset.csv`. A complete reproducible build script is available at `ml/build_raw_dataset.py`.

---

## 5. Feature Engineering Registry (60 Dimensions)

Features are computed exclusively from the URL text structure, domain registration metadata, character distributions, and lexical patterns. No feature is included during training that cannot be computed in real-time during live inference.

### A. URL Length & Structural Metrics
- `url_length`: Total character count of full URL string.
- `hostname_length`: Character length of the hostname/netloc.
- `path_length`: Length of the resource path string.
- `query_length`: Length of parameter query string.
- `domain_length`: Length of registered domain name.
- `num_dots`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_questionmarks`, `num_equal`, `num_at`, `num_ampersand`, `num_percent`, `num_tilde`, `num_plus`, `num_exclamation`, `num_dollar`, `num_hashtag`.
- `num_digits`: Absolute count of numeric digits.
- `digit_ratio`: Ratio of digits to total URL length.
- `num_subdirectories`: Count of path directory segments.
- `num_params`: Count of individual URL query parameters.

### B. Protocol & Deception Flags
- `is_https`: Binary indicator (1/0) of TLS encryption (evaluated as a feature, not safety proof).
- `has_ip_address`: Binary flag detecting raw IPv4 or IPv6 addresses in hostname.
- `has_port`: Binary flag detecting explicit non-standard network ports.
- `has_double_slash_in_path`: Detects `//` inside path (redirect obfuscation).
- `has_http_in_path_or_host`: Detects embedded `http` tokens in path (e.g. fake reverse proxies).
- `is_punycode`: Detects `xn--` IDN homograph attack patterns.
- `is_suspicious_tld`: Identifies registrars with documented high abuse rates (`.xyz`, `.top`, `.icu`, etc.).
- `has_client_server_in_domain`: Detects keywords like `server`, `client`, `admin` embedded in domain string.

### C. Suspicious Keyword Indicators
Scans for targeted high-risk keywords: `login`, `verify`, `account`, `password`, `secure`, `update`, `confirm`, `banking`, `signin`, `auth`, `admin`, `wallet`, `ebayisapi`, `webscr`, `token`, `recover`, `service`, `portal`, `alert`, `checkout`, `pay`.
- `suspicious_keyword_count`: Total occurrences of security keywords.
- `has_suspicious_keyword`: Binary presence flag.

### D. Statistical Entropy & Categorical TLDs
- `domain_entropy`: Shannon entropy $H(X) = -\sum P(x) \log_2 P(x)$ of registered domain (detects DGA domains).
- `url_entropy`: Shannon entropy of full URL (detects token hiding and payload encoding).
- `tld`: Top-22 One-Hot Encoded Top-Level Domains (`com`, `org`, `net`, `edu`, `gov`, `io`, `co`, `ru`, `xyz`, `top`, `de`, `uk`, `in`, `cn`, `app`, `info`, `biz`, `online`, `site`, `club`, `icu`, `none`) with automatic mapping of rare/unseen TLDs to `other`.

---

## 6. Machine Learning Methodology & Evaluation Benchmark

### Data Splitting Strategy
To guarantee unbiased evaluation and eliminate data leakage:
- **Training Partition (70%, 21,416 samples)**: Used exclusively to fit the preprocessor and train candidate classifiers.
- **Validation Partition (15%, 4,590 samples)**: Used for hyperparameter tuning and cross-validation.
- **Held-Out Test Partition (15%, 4,590 samples)**: Held strictly unseen until final model evaluation.
- **Stratification**: Preserved exact 50/50 class balance across all three splits.

### Multi-Model Comparison on Unseen Test Split

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate | False Negative Rate | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | 93.90% | 95.49% | 92.16% | 0.9379 | 0.9787 | 4.36% | 7.84% | 0.49s |
| **Random Forest** (Ensemble) | 97.56% | 97.85% | 97.25% | 0.9755 | 0.9976 | 2.14% | 2.75% | 0.71s |
| **XGBoost Classifier** (Selected Best) | **98.65%** | **99.08%** | **98.21%** | **0.9864** | **0.9990** | **0.92%** | **1.79%** | **0.68s** |

### Selected Best Model: XGBoost Classifier
**Rationale**: XGBoost achieved the highest F1-Score (**0.9864**) and ROC-AUC (**0.9990**) with an extremely low False Positive Rate (**0.92%**) and False Negative Rate (**1.79%**), ensuring minimal disruption for benign websites while catching 98.21% of malicious URLs.

### Confusion Matrix on Unseen Test Set (4,590 Samples)

$$\begin{pmatrix} \text{True Negatives (TN)} = 2,274 & \text{False Positives (FP)} = 21 \\ \text{False Negatives (FN)} = 41 & \text{True Positives (TP)} = 2,254 \end{pmatrix}$$

- **True Negatives (TN)**: 2,274 correctly classified benign URLs.
- **False Positives (FP)**: 21 benign URLs flagged as suspicious (False alarm rate: 0.92%).
- **False Negatives (FN)**: 41 malicious URLs missed (Miss rate: 1.79%).
- **True Positives (TP)**: 2,254 accurately intercepted threats.

---

## 7. Explainability Layer (SHAP TreeExplainer)

Rather than outputting opaque classifications, the system computes local Shapley values:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f_x(S \cup \{i\}) - f_x(S)]$$

Every URL scan returns the top positive and negative contributing factors mapped to human-readable cybersecurity descriptions:
- *Example Positive Factor*: `[+1.991] Long hostname (25 chars) exceeding standard legitimate domain conventions.`
- *Example Positive Factor*: `[+1.181] Top-Level Domain belongs to high-abuse/low-reputation registrar categories.`
- *Example Negative Factor*: `[-0.778] Registered domain matches high-reputation structural conventions.`

---

## 8. Calibrated Risk Representation & Recommendations

The model probability $P(\text{Malicious} \mid X) \in [0.0, 1.0]$ is converted into an integer Risk Index $(0 - 100)$:

```
0 ------------ 39 ---------------- 69 -------------- 100
[    LOW RISK    ] [   MEDIUM RISK    ] [    HIGH RISK    ]
```

### Risk Tier Definitions & Defensive Guidance

| Tier | Score Range | Prediction Label | User Recommendation |
| :--- | :---: | :--- | :--- |
| **LOW** | 0 – 39 | *Legitimate / Benign Pattern* | Low-risk characteristics detected based on structural and domain indicators. Continue with normal browsing precautions. Always verify SSL certificate indicators in your browser. |
| **MEDIUM** | 40 – 69 | *Suspicious / Anomalous Structure* | Moderate risk patterns and anomalies observed in URL composition. Exercise elevated caution: verify the organization identity before submitting passwords, credentials, or personal information. |
| **HIGH** | 70 – 100 | *Potentially Malicious / Phishing* | High-risk markers and deception signatures detected consistent with phishing or malicious infrastructure. Do NOT enter passwords, financial details, or download files from this link. |

---

## 9. Project Structure

```
ML_project/
├── data/
│   ├── raw/
│   │   ├── raw_urls_dataset.csv     # 30,596 benchmark dataset
│   │   └── dataset.csv              # Vrbančič raw reference dataset
│   ├── processed/
│   │   ├── train.csv                # 70% stratified training split
│   │   ├── val.csv                  # 15% validation split
│   │   └── test.csv                 # 15% unseen test split
│   ├── cyber_monitor.db             # SQLite persistent scan history
│   └── dataset_info.json            # Dataset provenance & class balance summary
├── ml/
│   ├── __init__.py
│   ├── build_raw_dataset.py         # Automated multi-feed dataset builder
│   ├── inspect_dataset.py           # Exploratory data analysis & verification tool
│   ├── feature_extractor.py         # 60-dimensional URL feature extraction engine
│   ├── preprocessor.py              # Preprocessing pipeline (StandardScaler + OneHot)
│   ├── train.py                     # Multi-model training, tuning & evaluation script
│   ├── explainer.py                 # SHAP TreeExplainer & local attribution module
│   ├── risk_scorer.py               # Calibrated risk scoring & recommendation engine
│   └── curated_patterns.py          # Supplementary authentic patterns
├── models/
│   ├── best_model.joblib            # Serialized best classifier (XGBoost)
│   ├── preprocessor.joblib          # Serialized preprocessor pipeline
│   ├── explainer.joblib             # Serialized SHAP TreeExplainer
│   ├── model_metadata.json          # Architecture, parameters, split sizes, metrics
│   └── evaluation_results.json      # Comparison table, confusion matrices & ROC curve
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application entrypoint & static mount
│   ├── config.py                    # App configuration, paths, and timeouts
│   ├── schemas.py                   # Pydantic validation schemas
│   ├── database.py                  # SQLite database persistence layer
│   ├── routes/
│   │   ├── scan.py                  # POST /api/scan & POST /api/batch-scan
│   │   ├── history.py               # GET /api/history & DELETE /api/history
│   │   ├── model_info.py            # GET /api/model-info & GET /api/dataset-info
│   │   └── health.py                # GET /api/health
│   └── services/
│       ├── scan_service.py          # Core inference orchestration service
│       └── network_inspector.py     # Safe HTTP HEAD probing with 2.5s timeout
├── frontend/
│   ├── index.html                   # Modern responsive cybersecurity dashboard
│   ├── css/
│   │   └── styles.css               # Glassmorphism dark-mode stylesheet
│   └── js/
│       ├── api.js                   # Backend API communication client
│       ├── charts.js                # Canvas ROC Curve & Feature Importance engine
│       └── app.js                   # Application state, scan handler & history manager
├── tests/
│   ├── test_feature_extractor.py    # Unit tests for URL normalization & feature metrics
│   ├── test_preprocessor.py         # Unit tests for preprocessing & unseen TLD handling
│   ├── test_model.py                # Unit tests for model inference & SHAP attribution
│   ├── test_risk_scorer.py          # Unit tests for threshold boundaries & advisories
│   ├── test_api.py                  # Integration tests for FastAPI endpoints
│   └── test_edge_cases.py           # Boundary tests (IPv4, IPv6, huge queries, punycode)
├── requirements.txt                 # Project dependencies
├── conftest.py                      # Pytest configuration
├── run.py                           # Single launcher for API + Dashboard
└── README.md                        # Complete project documentation
```

---

## 10. Installation & Quickstart

### Prerequisites
- Python **3.10+** (Tested on Python 3.13.5)
- Git & PowerShell / Terminal

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/your-username/ml-cyber-monitoring-system.git
cd ml-cyber-monitoring-system

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Dataset & Train Models (Reproducibility)
To regenerate the dataset from scratch and train candidate models:
```bash
# Fetch verified benchmark feeds (Tranco, PhishTank, URLhaus, OpenPhish)
python ml/build_raw_dataset.py

# Inspect dataset balance and verify 0 missing values
python ml/inspect_dataset.py

# Train Logistic Regression, Random Forest, and XGBoost; evaluate on test set
python ml/train.py
```

### 4. Run Test Suite
```bash
python -m pytest -v
```
*Expected result: 26 passed tests (100% success rate).*

### 5. Launch the Application
```bash
python run.py
```
Open your browser and navigate to:
- **Cybersecurity Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 11. REST API Specification

### 1. `POST /api/scan`
Analyzes a URL for cybersecurity risks.

**Request Payload:**
```json
{
  "url": "http://paypal.com.account-update.xyz/login.php?user=admin&token=928a",
  "include_network_probe": false
}
```

**Response Payload:**
```json
{
  "id": 1,
  "url": "http://paypal.com.account-update.xyz/login.php?user=admin&token=928a",
  "normalized_url": "http://paypal.com.account-update.xyz/login.php?user=admin&token=928a",
  "prediction": "Potentially Malicious / Phishing",
  "is_malicious": true,
  "riskLevel": "HIGH",
  "riskScore": 82,
  "modelProbability": 0.8192,
  "confidence": "63.8%",
  "color": "#EF4444",
  "badge": "high-risk",
  "keyFactors": [
    {
      "feature": "hostname_length",
      "description": "Long hostname (33 chars) exceeding standard legitimate domain conventions.",
      "raw_value": 33,
      "impact_score": 1.991,
      "direction": "increases_risk"
    },
    {
      "feature": "is_suspicious_tld",
      "description": "Top-Level Domain belongs to high-abuse/low-reputation registrar categories.",
      "raw_value": 1,
      "impact_score": 1.181,
      "direction": "increases_risk"
    }
  ],
  "extractedFeatures": {
    "url_length": 69,
    "is_https": 0,
    "num_dots": 3,
    "num_subdomains": 2,
    "has_ip_address": 0,
    "domain_entropy": 3.75,
    "tld": "xyz"
  },
  "recommendation": "High-risk markers and deception signatures detected consistent with phishing or malicious infrastructure. Do NOT enter passwords, financial details, or download files from this link.",
  "actionAdvice": "Avoid interaction; potential phishing or malicious host.",
  "explainabilityMethod": "SHAP TreeExplainer (Exact Game-Theoretic Shapley Values)",
  "executionTimeMs": 14.5,
  "timestamp": "2026-09-30T13:30:00.000000",
  "disclaimer": "ML-based probabilistic risk assessment — not a guarantee of website safety."
}
```

### 2. `GET /api/history`
Returns stored scan history with search, risk tier filtering, and pagination.

### 3. `GET /api/model-info`
Returns full benchmark model comparison, hyperparameters, confusion matrix, and feature importances.

---

## 12. Security, Privacy & Ethical Boundaries

1. **Zero Malware Execution**: The system parses URL strings lexically and statistically. It never executes untrusted JavaScript, downloads executable payloads, or renders suspicious iframes.
2. **Safe Probing**: Network requests use isolated `HEAD` requests with a hard 2.5-second timeout and certificate safety defaults.
3. **Zero Credential Harvesting**: The application never prompts for, stores, or transmits user passwords or personal data.
4. **Transparent Risk Communication**: The interface explicitly displays academic and statistical disclaimers, avoiding absolute safety guarantees.

---

## 13. Limitations & Future Scope

### Realistic Limitations
- **Evolving Attack Vectors**: Fast-flux DNS and dynamic zero-day obfuscations might not exhibit classic lexical anomalies.
- **Compromised Legitimate Domains**: If a reputable domain (e.g. `github.io` or `pages.dev`) is compromised, purely structural domain features may underestimate risk without real-time threat feed correlation.
- **Statistical Assessment**: Model predictions represent probabilistic likelihoods derived from training distributions, not cryptographic proofs.

### Future Scope
- **Real-Time Threat Intelligence Integration**: Connect VirusTotal, Google Safe Browsing, and AlienVault OTX APIs via async sidecars.
- **Browser Extension**: Build a Chromium / Firefox extension for passive background URL scoring.
- **Deep Learning Multimodal Inspection**: Incorporate lightweight visual DOM screenshot embedding models (CNN/ViT) to cross-verify visual brand identity against registered domain names.

---

## 14. Demonstration Guide (Faculty / Project Presentation)

To demonstrate this project to faculty or reviewers:
1. **Launch Server**: Run `python run.py` and open `http://127.0.0.1:8000/`.
2. **Live Threat Analysis**:
   - Click the `✓ Benign (Wikipedia)` chip $\rightarrow$ Observe **LOW RISK (0/100)** verdict with green advisory.
   - Click the `⚠ Phishing Deception` chip $\rightarrow$ Observe **HIGH RISK (82/100)** verdict with red alert, SHAP factor breakdown (`hostname_length`, `is_suspicious_tld`), and 60-feature telemetry.
3. **Demonstrate Explainability**: Explain how SHAP calculates exact Shapley values to pinpoint why the model assigned a specific threat level.
4. **Review Model Intelligence Tab**:
   - Present the **30,596** multi-source benchmark dataset.
   - Present the 3-model comparison table highlighting why XGBoost was chosen based on the test set F1-Score (**0.9864**) and low False Positive Rate (**0.92%**).
   - Review the interactive **ROC Curve ($AUC = 0.9990$)** and **Confusion Matrix** on the 4,590 test samples.
5. **Review Code Architecture**: Walk through modular components in `/ml`, `/backend`, and `/frontend`.

---

## 15. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
