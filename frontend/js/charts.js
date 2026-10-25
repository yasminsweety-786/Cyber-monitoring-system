/**
 * CyberCharts: Lightweight High-DPI HTML5 Canvas Chart Engine
 * Renders ROC curves, Horizontal Feature Importance bars, and Risk Distribution charts.
 */

const CyberCharts = {
  /**
   * Renders ROC Curve for Models
   */
  drawRocCurve(canvasId, rocData, modelName = "XGBoost (AUC: 0.9990)") {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    
    // Scale for high-DPI screens
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const w = rect.width;
    const h = rect.height;
    const padding = { top: 30, right: 30, bottom: 40, left: 45 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);

    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartH / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      const x = padding.left + (chartW / 5) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, h - padding.bottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Random Chance Baseline (Diagonal 45 deg)
    ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padding.left, h - padding.bottom);
    ctx.lineTo(w - padding.right, padding.top);
    ctx.stroke();

    // Model ROC Curve
    if (rocData && rocData.length > 0) {
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(6, 182, 212, 0.5)";
      ctx.shadowBlur = 10;
      ctx.beginPath();

      rocData.forEach((pt, idx) => {
        const px = padding.left + pt.fpr * chartW;
        const py = h - padding.bottom - pt.tpr * chartH;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    }

    // Axes Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("False Positive Rate (FPR)", padding.left + chartW / 2, h - 10);
    
    ctx.save();
    ctx.translate(15, padding.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("True Positive Rate (TPR / Recall)", 0, 0);
    ctx.restore();

    // Legend
    ctx.textAlign = "left";
    ctx.fillStyle = "#06b6d4";
    ctx.fillRect(padding.left + 10, padding.top + 10, 12, 4);
    ctx.fillText(modelName, padding.left + 30, padding.top + 15);
  },

  /**
   * Renders Feature Importance Horizontal Bars
   */
  drawFeatureImportances(canvasId, featuresList, maxItems = 10) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const items = (featuresList || []).slice(0, maxItems);
    if (items.length === 0) return;

    const maxVal = Math.max(...items.map(i => i.importance), 0.01);
    const rowH = (h - 20) / items.length;
    const labelW = 140;
    const barMaxW = w - labelW - 65;

    items.forEach((item, idx) => {
      const y = 15 + idx * rowH;
      const barW = (item.importance / maxVal) * barMaxW;

      // Label
      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      const cleanName = item.feature.length > 18 ? item.feature.substring(0, 16) + '..' : item.feature;
      ctx.fillText(cleanName, labelW - 10, y + rowH / 2 + 3);

      // Bar Background
      ctx.fillStyle = "rgba(51, 65, 85, 0.3)";
      ctx.beginPath();
      ctx.roundRect(labelW, y + 4, barMaxW, rowH - 8, 4);
      ctx.fill();

      // Bar Fill with Gradient
      const gradient = ctx.createLinearGradient(labelW, 0, labelW + barW, 0);
      gradient.addColorStop(0, "#3b82f6");
      gradient.addColorStop(1, "#06b6d4");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(labelW, y + 4, Math.max(barW, 4), rowH - 8, 4);
      ctx.fill();

      // Value label
      ctx.fillStyle = "#f8fafc";
      ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.fillText(item.importance.toFixed(3), labelW + barW + 8, y + rowH / 2 + 3);
    });
  }
};
