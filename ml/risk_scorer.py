"""
Risk Scoring and Recommendation Engine for Cyber Monitoring System.
Translates statistical model probabilities into tiered risk categories (LOW, MEDIUM, HIGH)
and provides cautious, actionable recommendations for end users and analysts.
"""

from typing import Dict, Any, Tuple

# Configurable risk thresholds
DEFAULT_LOW_THRESHOLD = 40     # 0 <= score < 40 -> LOW RISK
DEFAULT_HIGH_THRESHOLD = 70    # 70 <= score <= 100 -> HIGH RISK, in-between -> MEDIUM RISK

def calculate_risk_assessment(
    model_probability: float, 
    low_threshold: int = DEFAULT_LOW_THRESHOLD, 
    high_threshold: int = DEFAULT_HIGH_THRESHOLD
) -> Dict[str, Any]:
    """
    Converts ML model probability (0.0 to 1.0) into a calibrated risk score (0-100)
    and maps it to risk levels and defensive recommendations.
    
    Parameters:
    - model_probability: Probability of being malicious [0.0, 1.0] from the ML classifier.
    - low_threshold: Boundary below which risk is classified as LOW.
    - high_threshold: Boundary above which risk is classified as HIGH.
    
    Returns:
    - Dict with riskScore, riskLevel, rawProbability, confidence, color, badge, recommendation, and disclaimer.
    """
    # Clamp probability
    prob = max(0.0, min(1.0, float(model_probability)))
    
    # Risk score: 0 to 100
    risk_score = int(round(prob * 100))
    
    if risk_score < low_threshold:
        risk_level = "LOW"
        prediction_label = "Legitimate / Benign Pattern"
        color = "#10B981" # Emerald Green
        badge = "low-risk"
        recommendation = (
            "Low-risk characteristics detected based on structural and domain indicators. "
            "Proceed with standard web browsing precautions. Always verify SSL certificate indicators in your browser."
        )
        action_advice = "Safe to browse with standard awareness."
    elif risk_score < high_threshold:
        risk_level = "MEDIUM"
        prediction_label = "Suspicious / Anomalous Structure"
        color = "#F59E0B" # Amber / Warning
        badge = "medium-risk"
        recommendation = (
            "Moderate risk patterns and anomalies observed in URL composition. "
            "Exercise elevated caution: verify the organization identity before submitting passwords, credentials, or personal information."
        )
        action_advice = "Verify authenticity before submitting sensitive data."
    else:
        risk_level = "HIGH"
        prediction_label = "Potentially Malicious / Phishing"
        color = "#EF4444" # Red / Critical
        badge = "high-risk"
        recommendation = (
            "High-risk markers and deception signatures detected consistent with phishing or malicious infrastructure. "
            "Do NOT enter passwords, financial details, or download files from this link. Close or block this site."
        )
        action_advice = "Avoid interaction; potential phishing or malicious host."

    # Model confidence represents certainty in the assigned class
    confidence_pct = round(abs(prob - 0.5) * 2 * 100, 1) # 0% at 0.5 boundary, 100% at extremes (0.0 or 1.0)

    return {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "predictionLabel": prediction_label,
        "modelProbability": round(prob, 4),
        "confidence": f"{confidence_pct}%",
        "color": color,
        "badge": badge,
        "recommendation": recommendation,
        "actionAdvice": action_advice,
        "thresholds": {
            "low": low_threshold,
            "high": high_threshold
        },
        "disclaimer": "ML-based probabilistic risk assessment — not a guarantee of website safety."
    }

if __name__ == "__main__":
    for p in [0.05, 0.45, 0.88]:
        res = calculate_risk_assessment(p)
        print(f"Prob: {p} -> Score: {res['riskScore']} [{res['riskLevel']}] - Rec: {res['recommendation'][:60]}...")
