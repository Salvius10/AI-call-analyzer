from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_emotion_timeline(labelled_segments:list[dict])->list[dict]:
    """
    Runs VADER on CUSTOMER segments only.
    Returns: [{"time": float, "score": float, "label": str}, ...]
    Score: -1.0 (very negative) to +1.0 (very positive)
    """ 
    analyzer=SentimentIntensityAnalyzer()
    timeline=[]
    for seg in labelled_segments:
        if seg['speaker']!='CUSTOMER' or not seg['text']:
            continue
        scores=analyzer.polarity_scores(seg['text'])
        compound=scores['compound']

        if compound<=-0.5:
            label='Angry'
        elif compound<=-0.05:
            label='Frustrated'
        elif compound<=0.05:
            label='Neutral'
        elif compound<=0.5:
            label='Calm'
        else:
            label='Satisfied'
        
        timeline.append({
            'time':seg['start'],
            "score":compound,
            "label":label,
            "text":seg['text']
        })
    return timeline
