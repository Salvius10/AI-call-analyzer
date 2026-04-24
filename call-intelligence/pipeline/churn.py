from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Rule-based churn signals (no training data needed to start)
CHURN_KEYWORDS = ["cancel", "cancellation", "leave", "switching", "competitor",
                  "never again", "worst", "unacceptable", "ridiculous", "useless"]

COMPETITOR_KEYWORDS = ["competitor", "other company", "switching to", "going with"]

def compute_churn_risk(labeled_segments:list[dict],emotion_timeline:list[dict])->dict:
    """
    Returns:
        {"score": float (0-1), "risk_level": str, "signals": list[str]}
    """
    analyzer=SentimentIntensityAnalyzer()
    customer_text=" ".join(s['text'] for s in labeled_segments if s['speaker']=='CUSTOMER' and s['text'])
    signals=[]
    score=0.0
    churn_hits=[w for w in CHURN_KEYWORDS if w in customer_text]
    #keyword signals
    if churn_hits:
        score+=0.3
        signals.append(f"Used churn-related words: {', '.join(churn_hits)}")

    competitor_hits=[w for w in COMPETITOR_KEYWORDS if w in customer_text]
    if competitor_hits:
        score+=0.25
        signals.append("Mentioned competitor or switching")
    #frustration signals
    angry_count=sum(1 for e in emotion_timeline if e['label'] in ['Angry','Frustrated'])
    if angry_count>3:
        score+=0.2
        signals.append(f'High frustration sustained across {angry_count} segments')
