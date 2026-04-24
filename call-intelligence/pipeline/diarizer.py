import os 
from pyannote.audio import Pipeline

def diarize(audio_path:str)->list[dict]:
    """
    Returns:
        [{'speaker':'SPEAKER 00','start':float,'end':float},...]
    """
    hf_token=os.getenv("HF_TOKEN")
    pipeline=Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization=pipeline(audio_path)
    segments=[]
    for turn,_,speaker in diarization.itertracks(yield_labels=True):
        segments.append({
            'speaker':speaker,
            'start':round(turn.start,2),
            'end':round(turn.end,2)
        })
    return segments

def label_speakers(diarization:list[dict],transcript_segments:list[dict])->list[dict]:
    """
    Merges diarization with transcript segments.
    Assumes first speaker = AGENT, second speaker = CUSTOMER.
    Returns: [{"speaker": "AGENT"/"CUSTOMER", "start", "end", "text"}, ...]
    """
    # Map SPEAKER_00 → AGENT, SPEAKER_01 → CUSTOMER
    speaker_ids=list(dict.fromkeys(s['speaker'] for s in diarization))
    role_map={speaker_ids[0]:"AGENT"}
    if len(speaker_ids)>1:
        role_map[speaker_ids[1]]='CUSTOMER'
    labelled=[]
    for d in diarization:
        text_parts=[t['text'] for t in transcript_segments if t['start']>=d['start']-0.5 and t['stop']<=d['stop']]
        labelled.append({
            'speaker':role_map.get(d['speaker'],d['speaker']),
            'start':d['start'],
            'stop':d['stop'],
            'text':" ".join(text_parts).strip()
        })
    return labelled
    
