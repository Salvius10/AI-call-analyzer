import streamlit as st
import plotly.graph_objects as go
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Call Intelligence", layout="wide")
st.title("📞 Call Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload a call recording", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file:
    from pipeline.transcriber import transcribe
    from pipeline.diarizer import diarize, label_speakers
    from pipeline.emotion import get_emotion_timeline
    from pipeline.scorer import score_agent
    from pipeline.churn import compute_churn_risk
    from pipeline.reporter import generate_report

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("🎙️ Transcribing audio..."):
        transcript = transcribe(audio_path)

    with st.spinner("👥 Separating speakers..."):
        diarization = diarize(audio_path)
        labeled = label_speakers(diarization, transcript["segments"])

    with st.spinner("😤 Analyzing emotions..."):
        emotion_timeline = get_emotion_timeline(labeled)

    with st.spinner("📊 Scoring agent..."):
        agent_scores = score_agent(labeled)

    with st.spinner("⚠️ Computing churn risk..."):
        churn = compute_churn_risk(labeled, emotion_timeline)

    duration = max((s["end"] for s in labeled), default=0)

    with st.spinner("📝 Generating report..."):
        report = generate_report(labeled, agent_scores, churn, emotion_timeline, duration)

    # --- Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Transcript")
        for seg in labeled:
            icon = "🧑‍💼" if seg["speaker"] == "AGENT" else "👤"
            st.markdown(f"**{icon} {seg['speaker']}** `[{seg['start']}s]` {seg['text']}")

    with col2:
        st.subheader("👥 Speaker Breakdown")
        agent_turns = sum(1 for s in labeled if s["speaker"] == "AGENT")
        customer_turns = sum(1 for s in labeled if s["speaker"] == "CUSTOMER")
        st.metric("Agent turns", agent_turns)
        st.metric("Customer turns", customer_turns)

    st.subheader("😤 Customer Emotion Timeline")
    if emotion_timeline:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[e["time"] for e in emotion_timeline],
            y=[e["score"] for e in emotion_timeline],
            mode="lines+markers",
            text=[f"{e['label']}: {e['text'][:50]}" for e in emotion_timeline],
            hoverinfo="text+x+y",
            line=dict(color="crimson", width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏆 Agent Performance")
        st.metric("Overall Score", f"{agent_scores['overall']} / 100")
        for key in ["empathy", "problem_solving", "communication", "resolution_speed", "professionalism"]:
            st.progress(agent_scores[key] / 10, text=f"{key.replace('_', ' ').title()}: {agent_scores[key]}/10")
        st.info(f"✅ **Strength:** {agent_scores['strengths']}")
        st.warning(f"⚠️ **Improve:** {agent_scores['improvements']}")

    with col4:
        st.subheader("⚠️ Churn Risk")
        risk_score_pct = int(churn["score"] * 100)
        st.metric("Risk Score", f"{risk_score_pct}%", delta=churn["risk_level"])
        st.progress(churn["score"])
        st.markdown("**Signals detected:**")
        for signal in churn["signals"]:
            st.markdown(f"- {signal}")

    st.subheader("📋 Full Intelligence Report")
    st.markdown(report)

    os.unlink(audio_path)