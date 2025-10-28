#############################################
# AgriBot — LLM Vision + Local Tools
#############################################

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
from gtts import gTTS

# Optional mic
try:
    from streamlit_mic_recorder import speech_to_text
    HAVE_MIC = True
except:
    HAVE_MIC = False

# Must be first
st.set_page_config(page_title="AgriBot - LLM Vision", layout="wide")

#############################################################
# Text-to-speech
#############################################################
def tts_bytes(text):
    try:
        tts = gTTS(text=text, lang='en')
        bio = io.BytesIO()
        tts.write_to_fp(bio)
        bio.seek(0)
        return bio.read()
    except:
        return None


#############################################################
# Local ML — Soil health (simple model)
#############################################################
def soil_health_score(N, P, K, ph, moisture):
    score = N + P + K
    if ph < 5.5 or ph > 8.2:
        score -= 40
    if moisture < 25:
        score -= 25
    if score < 120:
        return "Low Fertility"
    elif score < 260:
        return "Moderate Fertility"
    return "High Fertility"


#############################################################
# Local ML — Irrigation Advisor
#############################################################
def irrigation_adv(temp, humidity, rain, soil_m):
    need = 40 + (temp - 20) * 2
    need -= rain * 0.7
    need -= soil_m * 0.5
    need -= humidity * 0.12
    return round(max(5, min(120, need)), 1)


#############################################################
# UI START
#############################################################

st.title("🌱 AgriBot — Agriculture Assistant")

api_key = st.sidebar.text_input("Gemini API Key (required for chat & image)", type="password")

st.sidebar.markdown("### Farming Tips")
st.sidebar.write("""
✅ Rotate crops  
✅ Monitor pests regularly  
✅ Use organic fertilizers  
✅ Adjust irrigation to weather  
""")

if api_key:
    genai.configure(api_key=api_key)

tab1, tab2, tab3 = st.tabs(["📸 Leaf Diagnosis (LLM Vision)", "🧪 Soil & Irrigation", "💬 Chat"])

#############################################################
# TAB 1 — Vision Diagnosis
#############################################################
with tab1:
    st.subheader("Upload crop leaf for disease detection (LLM Vision)")

    img = st.file_uploader("Upload leaf image", type=['png', 'jpg', 'jpeg'])

    if img and api_key:
        st.image(img)
        
        if st.button("Analyze with LLM"):
            llm = genai.GenerativeModel("gemini-2.5-pro")

            prompt = """You are an expert plant pathologist.
Inspect this leaf image and return ONLY:

**TABLE**
Include:
- Disease / Condition Name
- Severity Level (1-10)
- Possible Cause
- Visible Symptoms
- Recommended Treatment

**SUMMARY**
- Max 5 bullets
- Actionable instructions
- Keep concise
"""

            response = llm.generate_content([prompt, {"mime_type": img.type, "data": img.read()}])
            text = response.text

            st.markdown(text)

            audio = tts_bytes(text)
            if audio:
                st.audio(audio, format="audio/mp3")


#############################################################
# TAB 2 — Local ML Tools
#############################################################
with tab2:
    st.subheader("🧪 Soil Health (Offline Rule Model)")
    
    N = st.number_input("Nitrogen", 0, 200, 30)
    P = st.number_input("Phosphorus", 0, 200, 20)
    K = st.number_input("Potassium", 0, 200, 20)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
    moisture = st.number_input("Moisture %", 0, 100, 35)

    if st.button("Check Soil"):
        s = soil_health_score(N, P, K, ph, moisture)
        st.info(f"Soil status: {s}")

    st.markdown("---")

    st.subheader("💧 Irrigation Advice (Offline Rule Model)")
    temp = st.number_input("Temperature °C", 0.0, 50.0, 28.0)
    hum = st.number_input("Humidity %", 0, 100, 55)
    rain = st.number_input("Recent Rainfall (mm)", 0.0, 300.0, 0.0)
    soil_m = st.number_input("Soil Moisture %", 0.0, 100.0, 25.0)

    if st.button("Recommend Water"):
        liters = irrigation_adv(temp, hum, rain, soil_m)
        st.success(f"Recommended irrigation: ~{liters} L/acre")


#############################################################
# TAB 3 — Chat assistant
#############################################################
with tab3:
    st.subheader("Ask anything about farming")

    query_type = st.selectbox("Query Type",
        ["Crop Advice", "Pest Management", "Soil Health", "Weather Tips"])

    mode = st.radio("Input Type", ["Text", "Voice"] if HAVE_MIC else ["Text"])

    user_text = None
    if mode == "Text":
        user_text = st.chat_input("Ask here...")
    else:
        txt = speech_to_text(language='en')
        if txt:
            user_text = txt
            st.success(f"You said: {txt}")

    if user_text and api_key:
        st.chat_message("user").markdown(user_text)

        system = """
You are an expert agronomist.
Always reply with:

**TABLE**
Columns must be relevant based on topic.

**SUMMARY**
- max 6 bullets
- actionable tips
- concise
"""

        chat_model = genai.GenerativeModel("gemini-2.5-pro", system_instruction=system)
        resp = chat_model.start_chat().send_message(f"[{query_type}] {user_text}")
        text = resp.text

        st.chat_message("assistant").markdown(text)

        audio = tts_bytes(text)
        if audio:
            st.audio(audio, format="audio/mp3")
