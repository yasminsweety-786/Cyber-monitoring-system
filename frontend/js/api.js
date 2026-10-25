/**
 * CyberGuard API Service
 * Handles communication with the FastAPI cybersecurity backend.
 */

const API_BASE_URL = (window.location.origin && window.location.origin.startsWith('http'))
  ? window.location.origin 
  : 'http://127.0.0.1:8001';

const CyberAPI = {
  /**
   * Health check endpoint
   */
  async getHealth() {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/health`);
      return await resp.json();
    } catch (e) {
      console.warn("Backend offline or unreachable:", e);
      return { status: "offline", model_loaded: false };
    }
  },

  /**
   * Scans a single website URL
   */
  async scanUrl(url, includeNetworkProbe = false) {
    const resp = await fetch(`${API_BASE_URL}/api/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: url.trim(),
        include_network_probe: includeNetworkProbe
      })
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Network request failed" }));
      throw new Error(err.detail || "URL analysis failed");
    }

    return await resp.json();
  },

  /**
   * Fetches scan history with filters and pagination
   */
  async getHistory(search = "", riskLevel = "ALL", page = 1, pageSize = 20) {
    const params = new URLSearchParams();
    if (search) params.append("search", search);
    if (riskLevel && riskLevel !== "ALL") params.append("risk_level", riskLevel);
    params.append("page", page);
    params.append("page_size", pageSize);

    const resp = await fetch(`${API_BASE_URL}/api/history?${params.toString()}`);
    if (!resp.ok) throw new Error("Failed to load scan history");
    return await resp.json();
  },

  /**
   * Clears all scan history
   */
  async clearHistory() {
    const resp = await fetch(`${API_BASE_URL}/api/history`, { method: "DELETE" });
    if (!resp.ok) throw new Error("Failed to clear scan history");
    return await resp.json();
  },

  /**
   * Deletes a single scan history record
   */
  async deleteHistoryRecord(id) {
    const resp = await fetch(`${API_BASE_URL}/api/history/${id}`, { method: "DELETE" });
    if (!resp.ok) throw new Error("Failed to delete record");
    return await resp.json();
  },

  /**
   * Fetches model training metadata and multi-model evaluation metrics
   */
  async getModelInfo() {
    const resp = await fetch(`${API_BASE_URL}/api/model-info`);
    if (!resp.ok) throw new Error("Failed to load model intelligence");
    return await resp.json();
  },

  /**
   * Fetches dataset provenance and statistics
   */
  async getDatasetInfo() {
    const resp = await fetch(`${API_BASE_URL}/api/dataset-info`);
    if (!resp.ok) throw new Error("Failed to load dataset details");
    return await resp.json();
  },

  /**
   * Fetches feature list definitions
   */
  async getFeaturesList() {
    const resp = await fetch(`${API_BASE_URL}/api/features-list`);
    if (!resp.ok) throw new Error("Failed to load features list");
    return await resp.json();
  }
};
