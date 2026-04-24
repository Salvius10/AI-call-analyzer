import whisper

def transcribe(audio_path:str)->dict:
    """
    Returns:
        {
            "text":full transcript string,
            "segments":[{"start":float,"stop":float,"text":str},...]
        }
    """
    model=whisper.load_model('base')
    result=model.transcribe(audio_path)
    return {
        "text":result['text'],
        "segment":result['segments']
    }