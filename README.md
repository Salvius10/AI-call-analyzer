# 📞 Call Intelligence Dashboard

An AI-powered web app that analyzes customer service call recordings and generates a complete intelligence report — automatically, in seconds.

## 🚀 What It Does

Upload a call recording and the system will:

- **Transcribe** the full conversation using Groq Whisper API
- **Separate speakers** into Agent vs Customer using PyAnnote
- **Map customer emotions** over time with a live sentiment chart
- **Score the agent** on 5 performance dimensions
- **Predict churn risk** based on keywords, tone, and sentiment
- **Generate a full report** with actionable next steps using Groq LLaMA

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────┐
│  Upload Call Recording                  │
│  [Drag and drop audio file here]        │
├─────────────────────────────────────────┤
│  TRANSCRIPTION          SPEAKERS        │
│  Full call text    │  Agent vs Customer │
├─────────────────────────────────────────┤
│  EMOTION TIMELINE CHART                 │
│  (line chart showing sentiment over     │
│   time with timestamps)                 │
├─────────────────────────────────────────┤
│  AGENT SCORE    │    CHURN RISK         │
│  82 / 100       │    0% LOW RISK        │
├─────────────────────────────────────────┤
│  FULL INTELLIGENCE REPORT               │
│  (AI generated readable summary)        │
└─────────────────────────────────────────┘
```

## 🧠 Tech Stack

| Tool | Role |
|------|------|
| Groq Whisper API | Audio → Text transcription |
| PyAnnote | Speaker diarization (Agent vs Customer) |
| VADER | Text sentiment scoring |
| Groq LLaMA 3.3 | Agent scoring + report generation |
| Plotly | Emotion timeline chart |
| Streamlit | Dashboard UI |

## 📁 Project Structure

```
call-intelligence/
├── app.py                  # Streamlit dashboard
├── pipeline/
│   ├── transcriber.py      # Groq Whisper transcription
│   ├── diarizer.py         # PyAnnote speaker separation
│   ├── emotion.py          # VADER sentiment analysis
│   ├── scorer.py           # Groq LLaMA agent scoring
│   ├── churn.py            # Rule-based churn risk
│   └── reporter.py         # Groq LLaMA report generation
├── .env                    # API keys (never commit this)
├── .gitignore
└── requirements.txt
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/call-intelligence.git
cd call-intelligence
```

### 2. Create a virtual environment

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**Getting your keys:**
- **Groq API key** → [console.groq.com](https://console.groq.com) → Sign up → API Keys
- **HuggingFace token** → [huggingface.co](https://huggingface.co) → Settings → Access Tokens → Read access

**Accept PyAnnote model terms** (required):
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### 5. Install FFmpeg

**Windows:**
```bash
winget install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📊 Sample Output

**Agent Performance:**
- Empathy: 6/10
- Problem Solving: 8/10
- Communication: 9/10
- Resolution Speed: 7/10
- Professionalism: 9/10
- **Overall: 82/100**

**Churn Risk:** 0% LOW RISK 🟢

**Report excerpt:**
> The call lasted 1 minute and 58 seconds, during which the customer inquired about updating the map in their car. The agent provided a clear explanation of the benefits of the update and successfully set up an order...

## 🔌 MERN Integration

This pipeline can be exposed as a REST API using FastAPI and called from a Node.js/Express backend. See the integration guide in the docs.

## 📝 Supported Audio Formats

- MP3
- WAV
- M4A
- OGG

## ⚠️ Notes

- First run takes longer as PyAnnote downloads model weights
- Subsequent runs are faster due to model caching
- Runs on CPU — no GPU required
- Tested on Python 3.11

