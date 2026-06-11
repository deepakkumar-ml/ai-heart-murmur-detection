import streamlit as st
import os
import glob
import tempfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CardioAI Pro",
    page_icon="🫀",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#0B1220,
#111827,
#0F172A
);
}

[data-testid="stSidebar"]{
background:#111827;
}

.hero{
padding:30px;
border-radius:20px;
background:linear-gradient(
135deg,
rgba(6,182,212,.15),
rgba(139,92,246,.15)
);
border:1px solid rgba(255,255,255,.1);
margin-bottom:20px;
}

.stButton > button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
border:none;
border-radius:12px;
background:linear-gradient(
135deg,
#06B6D4,
#8B5CF6
);
color:white;
}

.metric-box{
padding:15px;
border-radius:15px;
background:rgba(255,255,255,.05);
border:1px solid rgba(255,255,255,.08);
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================

MODEL_FOLDER = "Model files"

CLASSES = [
    "Artifact (Noise)",
    "Murmur (Heart Defect)",
    "Normal (Healthy)"
]

DURATION = 10
SAMPLE_RATE = 22050
FEATURES = 52

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_heart_model():

    files = (
        glob.glob(os.path.join(MODEL_FOLDER, "*.keras"))
        +
        glob.glob(os.path.join(MODEL_FOLDER, "*.h5"))
    )

    if not files:
        return None, None

    model = load_model(files[0])

    return model, os.path.basename(files[0])

model, model_name = load_heart_model()

# =====================================================
# PREPROCESS
# =====================================================

def preprocess_audio(file_path):

    try:

        X, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        input_length = SAMPLE_RATE * DURATION

        dur = librosa.get_duration(
            y=X,
            sr=sr
        )

        if round(dur) < DURATION:
            X = librosa.util.fix_length(
                data=X,
                size=input_length
            )

        mfccs = np.mean(
            librosa.feature.mfcc(
                y=X,
                sr=sr,
                n_mfcc=FEATURES
            ).T,
            axis=0
        )

        final_input = np.reshape(
            mfccs,
            (1, FEATURES, 1)
        )

        return final_input, X, sr

    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🫀 CardioAI Pro")

    st.markdown("---")

    if model:

        st.success(
            f"✅ Model Loaded\n\n{model_name}"
        )

    else:

        st.error(
            "❌ Model Not Found"
        )

    st.markdown("---")

    st.info("""
Upload a heart sound WAV file.

System will:

• Extract MFCC Features

• Run LSTM Analysis

• Generate Diagnosis
""")

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">

<h1>🩺 CardioAI Pro</h1>

<p>
AI Powered Heart Murmur Detection System
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD METRICS
# =====================================================

c1,c2,c3,c4 = st.columns(4)

c1.metric("Model","LSTM")
c2.metric("Features","52 MFCC")
c3.metric("Window","10 sec")
c4.metric("Status","Online")

st.markdown("### 👤 Patient Information")

p1,p2,p3 = st.columns(3)

patient_name = p1.text_input("Patient Name")
patient_age = p2.number_input("Age",1,120,25)
patient_gender = p3.selectbox(
    "Gender",
    ["Male","Female","Other"]
)

st.markdown("---")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    input_data, signal, sr = preprocess_audio(
        temp_path
    )

    # =============================================
    # VISUALIZATION SECTION
    # =============================================

    left,right = st.columns([2,1])

    with left:

        st.subheader("📈 Waveform")

        fig, ax = plt.subplots(
            figsize=(12,3)
        )

        ax.plot(signal)

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

        st.subheader("🌈 Spectrogram")

        fig2, ax2 = plt.subplots(
            figsize=(12,4)
        )

        D = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(signal)
            ),
            ref=np.max
        )

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax2
        )

        fig2.colorbar(
            img,
            ax=ax2,
            format="%+2.0f dB"
        )

        st.pyplot(fig2)

    with right:

        st.subheader("⚙ Analysis")

        if st.button(
            "🚀 Run AI Diagnosis"
        ):

            prediction = model.predict(
                input_data,
                verbose=0
            )

            probs = prediction[0]

            winner_index = np.argmax(probs)

            confidence = probs[winner_index]

            label = CLASSES[winner_index]

            st.session_state["probs"] = probs
            st.session_state["confidence"] = confidence
            st.session_state["label"] = label

# =====================================================
# RESULTS
# =====================================================

if "probs" in st.session_state:

    probs = st.session_state["probs"]

    confidence = st.session_state["confidence"]

    label = st.session_state["label"]

    st.markdown("---")

    st.header("🩺 Diagnosis Report")

    if "Murmur" in label:

        st.error(
            f"🔴 {label}\n\nConfidence: {confidence*100:.2f}%"
        )

    elif "Normal" in label:

        st.success(
            f"🟢 {label}\n\nConfidence: {confidence*100:.2f}%"
        )

    else:

        st.warning(
            f"⚠ {label}\n\nConfidence: {confidence*100:.2f}%"
        )

    col1,col2 = st.columns(2)

    with col1:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(confidence*100),
                title={"text":"AI Confidence"},
                gauge={
                    "axis":{
                        "range":[0,100]
                    }
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    with col2:

        df = pd.DataFrame({
            "Condition":[
                "Artifact",
                "Murmur",
                "Normal"
            ],
            "Probability":probs
        })

        pie = px.pie(
            df,
            names="Condition",
            values="Probability",
            hole=0.55
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    st.subheader("📋 Recommendations")

    if "Normal" in label:

        st.success("""
• Heart rhythm appears normal

• Continue routine monitoring

• Annual cardiac screening advised
""")

    elif "Murmur" in label:

        st.error("""
• Possible murmur detected

• Consult a cardiologist

• Echocardiogram recommended

• Further clinical evaluation advised
""")

    else:

        st.warning("""
• Audio contains excessive noise

• Re-record sample

• Reduce background interference

• Verify stethoscope placement
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "CardioAI Pro • TensorFlow • Librosa • Streamlit"
)