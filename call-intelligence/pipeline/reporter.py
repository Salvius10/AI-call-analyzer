import os 
from groq import Groq

def generate_report(labeled_segments,agent_scores,churn_data,emotion_timeline,duration_seconds)->str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    minutes,seconds=divmod(int(duration_seconds),60)
    peak_frustration=min(emotion_timeline,key=lambda x:x['score'],default=None)
    context = f"""
    Call Duration: {minutes}m {seconds}s
    Agent Scores: {agent_scores}
    Churn Risk: {churn_data['risk_level']} ({int(churn_data['score']*100)}%)
    Churn Signals: {', '.join(churn_data['signals'])}
    Peak Frustration: at {peak_frustration['time']}s — "{peak_frustration['text']}" ({peak_frustration['label']})
    Transcript excerpt (first 2000 chars):
    {"".join(f"{s['speaker']}: {s['text']} " for s in labeled_segments)[:2000]}
    """

    prompt = f"""You are a call center intelligence system. Write a professional call intelligence report based on this data.
    {context}
    Format the report with these sections:
    1. CALL SUMMARY (2-3 sentences)
    2. AGENT PERFORMANCE (reference the scores, give specific feedback)
    3. CUSTOMER HEALTH & CHURN RISK (explain the risk and recommend action)
    4. RECOMMENDED NEXT STEPS (2-3 bullet points)
    Be specific, professional, and actionable. Keep it under 300 words."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()
