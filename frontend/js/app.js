/**
 * CyberGuard AI - Application Logic & Dashboard Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  App.init();
});

const App = {
  activeTab: "scanner",
  historyPage: 1,
  modelDataCache: null,

  init() {
    this.bindEvents();
    this.checkHealth();
    this.loadHistory();
    this.loadModelIntelligence();
  },

  bindEvents() {
    // Navigation tabs
    document.querySelectorAll(".nav-tab-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const tabId = e.currentTarget.dataset.tab;
        this.switchTab(tabId);
      });
    });

    // Scan form
    const scanBtn = document.getElementById("scan-btn");
    const urlInput = document.getElementById("url-input");
    
    if (scanBtn) {
      scanBtn.addEventListener("click", () => this.handleScan());
    }
    if (urlInput) {
      urlInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") this.handleScan();
      });
    }

    // Sample URL chips
    document.querySelectorAll(".sample-chip").forEach(chip => {
      chip.addEventListener("click", (e) => {
        const sampleUrl = e.currentTarget.dataset.url;
        if (urlInput) {
          urlInput.value = sampleUrl;
          this.handleScan();
        }
      });
    });

    // History controls
    const historySearch = document.getElementById("history-search");
    const riskFilter = document.getElementById("history-risk-filter");
    const clearHistoryBtn = document.getElementById("clear-history-btn");

    if (historySearch) {
      let debounceTimer;
      historySearch.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => this.loadHistory(), 300);
      });
    }

    if (riskFilter) {
      riskFilter.addEventListener("change", () => this.loadHistory());
    }

    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener("click", () => this.handleClearHistory());
    }
  },

  switchTab(tabId) {
    this.activeTab = tabId;
    
    // Update nav buttons
    document.querySelectorAll(".nav-tab-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.tab === tabId);
    });

    // Update tab panes
    document.querySelectorAll(".tab-pane").forEach(p => {
      p.classList.toggle("active", p.id === `tab-${tabId}`);
    });

    if (tabId === "history") {
      this.loadHistory();
    } else if (tabId === "model-info") {
      this.renderModelCharts();
    }
  },

  async checkHealth() {
    const statusText = document.getElementById("system-status-text");
    const statusDot = document.getElementById("system-status-dot");
    const data = await CyberAPI.getHealth();
    
    if (data.status === "healthy") {
      if (statusText) statusText.textContent = "AI Engine Online (v1.0)";
      if (statusDot) statusDot.style.backgroundColor = "var(--risk-low)";
    } else {
      if (statusText) statusText.textContent = "Engine Offline / Initializing";
      if (statusDot) statusDot.style.backgroundColor = "var(--risk-med)";
    }
  },

  async handleScan() {
    const urlInput = document.getElementById("url-input");
    const networkProbeCheckbox = document.getElementById("network-probe-checkbox");
    const progressCard = document.getElementById("pipeline-progress");
    const resultContainer = document.getElementById("scan-result");
    const scanBtn = document.getElementById("scan-btn");

    const rawUrl = urlInput ? urlInput.value.trim() : "";
    if (!rawUrl) {
      alert("Please enter a website URL to scan.");
      return;
    }

    const includeProbe = networkProbeCheckbox ? networkProbeCheckbox.checked : false;

    // Reset and show progress steps
    if (resultContainer) resultContainer.style.display = "none";
    if (progressCard) progressCard.style.display = "block";
    if (scanBtn) scanBtn.disabled = true;

    this.animatePipelineSteps(async () => {
      try {
        const scanResult = await CyberAPI.scanUrl(rawUrl, includeProbe);
        if (progressCard) progressCard.style.display = "none";
        this.renderScanResult(scanResult);
      } catch (err) {
        if (progressCard) progressCard.style.display = "none";
        alert(`Analysis failed: ${err.message}`);
      } finally {
        if (scanBtn) scanBtn.disabled = false;
      }
    });
  },

  animatePipelineSteps(onComplete) {
    const steps = ["step-normalize", "step-extract", "step-ml", "step-explain", "step-verdict"];
    let current = 0;

    steps.forEach(s => {
      const el = document.getElementById(s);
      if (el) {
        el.classList.remove("active", "completed");
      }
    });

    const interval = setInterval(() => {
      if (current > 0) {
        const prevEl = document.getElementById(steps[current - 1]);
        if (prevEl) {
          prevEl.classList.remove("active");
          prevEl.classList.add("completed");
        }
      }

      if (current < steps.length) {
        const curEl = document.getElementById(steps[current]);
        if (curEl) curEl.classList.add("active");
        current++;
      } else {
        clearInterval(interval);
        if (onComplete) onComplete();
      }
    }, 120);
  },

  renderScanResult(data) {
    const resultContainer = document.getElementById("scan-result");
    if (!resultContainer) return;
    resultContainer.style.display = "block";

    // 1. URL display
    const urlDisplay = document.getElementById("result-url-display");
    if (urlDisplay) urlDisplay.textContent = data.url;

    // 2. Risk Score & Gauge
    const scoreVal = document.getElementById("gauge-score-val");
    const gaugeFill = document.getElementById("gauge-fill-circle");
    const badge = document.getElementById("result-risk-badge");
    
    if (scoreVal) scoreVal.textContent = data.riskScore;

    // Gauge circle animation (circumference is ~502 for r=80)
    if (gaugeFill) {
      const circumference = 2 * Math.PI * 80;
      const offset = circumference - (data.riskScore / 100) * circumference;
      gaugeFill.style.strokeDashoffset = offset;
      gaugeFill.style.stroke = data.color;
    }

    if (badge) {
      badge.textContent = `${data.riskLevel} RISK`;
      badge.className = `risk-level-badge ${data.badge}`;
    }

    // 3. Metadata strip
    const probVal = document.getElementById("meta-prob-val");
    const confVal = document.getElementById("meta-conf-val");
    const timeVal = document.getElementById("meta-time-val");

    if (probVal) probVal.textContent = (data.modelProbability * 100).toFixed(1) + "%";
    if (confVal) confVal.textContent = data.confidence;
    if (timeVal) timeVal.textContent = data.executionTimeMs + " ms";

    // 4. Recommendation
    const recBox = document.getElementById("recommendation-card");
    const recTitle = document.getElementById("rec-title-text");
    const recDesc = document.getElementById("rec-desc-text");

    if (recBox) {
      recBox.className = `recommendation-box ${data.riskLevel.toLowerCase()}`;
    }
    if (recTitle) recTitle.textContent = data.prediction;
    if (recDesc) recDesc.textContent = data.recommendation;

    // 5. Explainability Factors (SHAP)
    const explainMethodLabel = document.getElementById("explain-method-label");
    if (explainMethodLabel) explainMethodLabel.textContent = data.explainabilityMethod;

    const factorsContainer = document.getElementById("factors-container");
    if (factorsContainer) {
      factorsContainer.innerHTML = "";
      (data.keyFactors || []).forEach(factor => {
        const isUp = factor.direction === "increases_risk";
        const card = document.createElement("div");
        card.className = "factor-card";
        card.innerHTML = `
          <div class="factor-info">
            <div class="factor-icon ${isUp ? 'risk-up' : 'risk-down'}">
              ${isUp ? '▲' : '▼'}
            </div>
            <div>
              <div class="factor-desc">${factor.description}</div>
              <span class="factor-feature-tag">${factor.feature}</span>
            </div>
          </div>
          <div class="factor-impact-bar-wrap">
            <span class="impact-score-num" style="color: ${isUp ? 'var(--risk-high)' : 'var(--risk-low)'}">
              ${isUp ? '+' : ''}${factor.impact_score.toFixed(3)}
            </span>
          </div>
        `;
        factorsContainer.appendChild(card);
      });
    }

    // 6. Extracted Features Inspector
    const featGrid = document.getElementById("features-inspector-grid");
    if (featGrid) {
      featGrid.innerHTML = "";
      const feats = data.extractedFeatures || {};
      const highlightKeys = [
        "url_length", "domain_length", "is_https", "num_dots", "num_subdomains",
        "has_ip_address", "suspicious_keyword_count", "domain_entropy", "url_entropy", "tld"
      ];

      highlightKeys.forEach(k => {
        if (k in feats) {
          const pill = document.createElement("div");
          pill.className = "feature-pill";
          pill.innerHTML = `
            <span class="feat-label">${k.replace(/_/g, ' ').toUpperCase()}</span>
            <span class="feat-val">${feats[k]}</span>
          `;
          featGrid.appendChild(pill);
        }
      });
    }

    // Refresh history silently
    this.loadHistory();
  },

  async loadHistory() {
    const searchInput = document.getElementById("history-search");
    const riskFilter = document.getElementById("history-risk-filter");
    const tableBody = document.getElementById("history-table-body");
    const emptyState = document.getElementById("history-empty-state");

    const search = searchInput ? searchInput.value.trim() : "";
    const risk = riskFilter ? riskFilter.value : "ALL";

    try {
      const data = await CyberAPI.getHistory(search, risk, this.historyPage, 20);

      // Update stat counter cards
      const totalCount = document.getElementById("stat-total-scans");
      const lowCount = document.getElementById("stat-low-scans");
      const medCount = document.getElementById("stat-med-scans");
      const highCount = document.getElementById("stat-high-scans");

      if (totalCount) totalCount.textContent = (data.low_risk_count + data.medium_risk_count + data.high_risk_count).toLocaleString();
      if (lowCount) lowCount.textContent = data.low_risk_count.toLocaleString();
      if (medCount) medCount.textContent = data.medium_risk_count.toLocaleString();
      if (highCount) highCount.textContent = data.high_risk_count.toLocaleString();

      if (!tableBody) return;
      tableBody.innerHTML = "";

      if (data.items.length === 0) {
        if (emptyState) emptyState.style.display = "block";
        return;
      }

      if (emptyState) emptyState.style.display = "none";

      data.items.forEach(item => {
        const row = document.createElement("tr");
        const badgeClass = item.risk_level.toLowerCase();
        row.innerHTML = `
          <td class="url-table-cell" title="${item.url}">${item.url}</td>
          <td><span class="badge-tag ${badgeClass}">${item.risk_level}</span></td>
          <td style="font-family: var(--font-mono); font-weight: 600;">${item.risk_score}</td>
          <td style="font-size: 0.8rem; color: var(--text-muted);">${item.key_factor_summary || 'Standard'}</td>
          <td style="font-size: 0.75rem; color: var(--text-muted);">${new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
          <td>
            <button class="btn-secondary" style="padding: 2px 8px; font-size: 0.75rem;" onclick="App.deleteHistoryItem(${item.id})">
              ✕
            </button>
          </td>
        `;
        tableBody.appendChild(row);
      });
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  },

  async deleteHistoryItem(id) {
    try {
      await CyberAPI.deleteHistoryRecord(id);
      this.loadHistory();
    } catch (e) {
      alert("Failed to delete record.");
    }
  },

  async handleClearHistory() {
    if (!confirm("Are you sure you want to clear all scan records?")) return;
    try {
      await CyberAPI.clearHistory();
      this.loadHistory();
    } catch (e) {
      alert("Failed to clear history.");
    }
  },

  async loadModelIntelligence() {
    try {
      const [modelInfo, datasetInfo] = await Promise.all([
        CyberAPI.getModelInfo(),
        CyberAPI.getDatasetInfo()
      ]);

      this.modelDataCache = { modelInfo, datasetInfo };
      this.renderModelIntelligenceUI(modelInfo, datasetInfo);
    } catch (e) {
      console.warn("Could not load model intelligence:", e);
    }
  },

  renderModelIntelligenceUI(modelInfo, datasetInfo) {
    // Dataset Summary Card
    const totalRecords = document.getElementById("ds-total-records");
    const legitCount = document.getElementById("ds-legit-count");
    const phishCount = document.getElementById("ds-phish-count");
    const featCount = document.getElementById("ds-feat-count");

    if (totalRecords) totalRecords.textContent = datasetInfo.total_records.toLocaleString();
    if (legitCount) legitCount.textContent = datasetInfo.class_distribution["0"].toLocaleString();
    if (phishCount) phishCount.textContent = datasetInfo.class_distribution["1"].toLocaleString();
    if (featCount) featCount.textContent = modelInfo.feature_count;

    // Multi-Model Comparison Table
    const compTableBody = document.getElementById("model-comparison-table-body");
    if (compTableBody && modelInfo.all_models_comparison) {
      compTableBody.innerHTML = "";
      Object.entries(modelInfo.all_models_comparison).forEach(([name, m]) => {
        const isSelected = name === modelInfo.model_name;
        const row = document.createElement("tr");
        if (isSelected) row.className = "selected-model";
        row.innerHTML = `
          <td>${name} ${isSelected ? '★ (Selected)' : ''}</td>
          <td>${(m.test.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.test.precision * 100).toFixed(2)}%</td>
          <td>${(m.test.recall * 100).toFixed(2)}%</td>
          <td>${m.test.f1_score.toFixed(4)}</td>
          <td>${m.test.roc_auc.toFixed(4)}</td>
          <td>${(m.test.false_positive_rate * 100).toFixed(2)}%</td>
          <td>${(m.test.false_negative_rate * 100).toFixed(2)}%</td>
          <td>${m.train_time_sec}s</td>
        `;
        compTableBody.appendChild(row);
      });
    }

    // Confusion Matrix
    const selectedMetrics = modelInfo.metrics || {};
    const cm = selectedMetrics.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 };
    
    const tnVal = document.getElementById("cm-tn-val");
    const fpVal = document.getElementById("cm-fp-val");
    const fnVal = document.getElementById("cm-fn-val");
    const tpVal = document.getElementById("cm-tp-val");

    if (tnVal) tnVal.textContent = cm.tn.toLocaleString();
    if (fpVal) fpVal.textContent = cm.fp.toLocaleString();
    if (fnVal) fnVal.textContent = cm.fn.toLocaleString();
    if (tpVal) tpVal.textContent = cm.tp.toLocaleString();

    this.renderModelCharts();
  },

  renderModelCharts() {
    if (!this.modelDataCache) return;
    const { modelInfo } = this.modelDataCache;

    // 1. Draw ROC Curve
    const selectedModel = modelInfo.all_models_comparison[modelInfo.model_name];
    if (selectedModel && selectedModel.test.roc_curve) {
      CyberCharts.drawRocCurve(
        "roc-curve-canvas", 
        selectedModel.test.roc_curve,
        `${modelInfo.model_name} (AUC: ${selectedModel.test.roc_auc.toFixed(4)})`
      );
    }

    // 2. Draw Feature Importance Bar Chart
    if (modelInfo.top_feature_importances) {
      CyberCharts.drawFeatureImportances(
        "feature-importance-canvas", 
        modelInfo.top_feature_importances, 
        10
      );
    }
  }
};
