from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CHURN_KEYWORDS = ["cancel", "cancellation", "leave", "switching", "competitor",
                  "never again", "worst", "unacceptable", "ridiculous", "useless"]

COMPETITOR_KEYWORDS = ["competitor", "other company", "switching to", "going with"]

def compute_churn_risk(labeled_segments: list[dict], emotion_timeline: list[dict]) -> dict:
    analyzer = SentimentIntensityAnalyzer()
    customer_text = " ".join(
        s["text"] for s in labeled_segments
        if s["speaker"] == "CUSTOMER" and s["text"]
    )

    signals = []
    score = 0.0

    # Keyword signals
    churn_hits = [w for w in CHURN_KEYWORDS if w in customer_text]
    if churn_hits:
        score += 0.3
        signals.append(f"Used churn-related words: {', '.join(churn_hits)}")

    competitor_hits = [w for w in COMPETITOR_KEYWORDS if w in customer_text]
    if competitor_hits:
        score += 0.25
        signals.append("Mentioned competitor or switching")

    # Frustration signals
    angry_count = sum(1 for e in emotion_timeline if e["label"] in ["Angry", "Frustrated"])
    if angry_count > 3:
        score += 0.2
        signals.append(f"High frustration sustained across {angry_count} segments")

    # Did call end negatively?
    if emotion_timeline:
        final_score = emotion_timeline[-1]["score"]
        if final_score < -0.2:
            score += 0.15
            signals.append("Call ended on a negative note")

    # Average customer sentiment
    avg_sentiment = (
        sum(e["score"] for e in emotion_timeline) / len(emotion_timeline)
        if emotion_timeline else 0
    )
    if avg_sentiment < -0.3:
        score += 0.1
        signals.append("Overall negative sentiment throughout call")

    score = min(score, 1.0)

    if score >= 0.6:
        risk_level = "HIGH RISK 🔴"
    elif score >= 0.35:
        risk_level = "MEDIUM RISK 🟡"
    else:
        risk_level = "LOW RISK 🟢"

    return {
        "score": round(score, 2),
        "risk_level": risk_level,
        "signals": signals if signals else ["No major churn signals detected"]
    }