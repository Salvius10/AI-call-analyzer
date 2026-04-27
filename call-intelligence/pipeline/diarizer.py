import os
import torch
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment


def diarize(audio_path: str) -> list[dict]:
    # Convert to wav using pydub
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = audio_path + "_converted.wav"
    audio.export(wav_path, format="wav")

    hf_token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    # Load audio as numpy array using pydub — no torchaudio needed
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(np.int16).max  # normalize to -1.0 to 1.0
    waveform = torch.tensor(samples).unsqueeze(0)  # shape: (1, samples)
    audio_input = {"waveform": waveform, "sample_rate": 16000}

    diarization = pipeline(audio_input)
    diarization = pipeline(audio_input)
    
    # Debug — let's see what we're working with
    
    os.unlink(wav_path)

    # Use speaker_diarization attribute — this is the Annotation object
    annotation = diarization.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return segments


def label_speakers(diarization: list[dict], transcript_segments: list[dict]) -> list[dict]:
    speaker_ids = list(dict.fromkeys(s["speaker"] for s in diarization))
    role_map = {speaker_ids[0]: "AGENT"}
    if len(speaker_ids) > 1:
        role_map[speaker_ids[1]] = "CUSTOMER"

    labeled = []
    for d in diarization:
        text_parts = [
            t["text"] for t in transcript_segments
            if t["start"] >= d["start"] - 0.5 and t["end"] <= d["end"] + 0.5
        ]
        labeled.append({
            "speaker": role_map.get(d["speaker"], d["speaker"]),
            "start": d["start"],
            "end": d["end"],
            "text": " ".join(text_parts).strip()
        })
    return labeled