import os 
from groq import Groq
import json

def score_agent(labeled_segments:list[dict])->dict:
    """
    Returns:
        {
            "empathy": int, "problem_solving": int, "communication": int,
            "resolution_speed": int, "professionalism": int, "overall": int,
            "strengths": str, "improvements": str
        }
    """
    
    client=Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    transcript = "\n".join(
        f"{s['speaker']} [{s['start']}s]: {s['text']}"
        for s in labeled_segments if s["text"]
    )
    prompt = f"""You are a call center quality analyst. Analyze this call transcript and score the AGENT.

    TRANSCRIPT:
    {transcript}

    Respond ONLY with a valid JSON object, no markdown, no explanation:
    {{
    "empathy": <1-10>,
    "problem_solving": <1-10>,
    "communication": <1-10>,
    "resolution_speed": <1-10>,
    "professionalism": <1-10>,
    "overall": <1-100>,
    "strengths": "<one sentence>",
    "improvements": "<one sentence>"
    }}"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)