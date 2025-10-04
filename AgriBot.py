import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64

# --- Streamlit UI ---
st.set_page_config(page_title="AgriBot - Farming Assistant", layout="wide")
st.title("🌱 AgriBot: Farming Assistant Chat")

# Sidebar: Gemini API Key
st.sidebar.header("Setup")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# Sidebar: Farming tips
st.sidebar.subheader("Farming Tips")
st.sidebar.markdown("""
- Rotate crops to maintain soil health. 
- Monitor pests regularly.
- Use organic fertilizers when possible.
- Check weather forecasts before irrigation.
""")

# --- Helper functions ---
def get_image_bytes(uploaded_file):
    if uploaded_file:
        image = Image.open(uploaded_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return None

def display_chat_history(chat_history):
    """Display chat messages using Streamlit chat interface"""
    for chat in chat_history:
        role = "user" if chat['role'] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(chat['message'])

# --- Main App Logic ---
if api_key:
    genai.configure(api_key=api_key)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'model' not in st.session_state:
        system_prompt = (
            "You are an expert agronomist providing helpful and actionable advice to farmers. "
            "If the user asks anything outside agriculture context, always reply with: "
            "'I can answer queries related to Agriculture.'"
        )
        st.session_state.model = genai.GenerativeModel(
            "gemini-2.5-pro",
            system_instruction=system_prompt
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])

    # Query type selection
    query_type = st.selectbox(
        "Select Query Type:",
        ["Crop Advice", "Pest Management", "Soil Health", "Weather Tips"]
    )

    # Chat input
    user_input = st.chat_input("Type your question here:")
    uploaded_file = st.file_uploader("Upload crop image (optional):", type=['png', 'jpg', 'jpeg'])

    if user_input:
        # Handle image upload
        image_data = get_image_bytes(uploaded_file)
        full_prompt = f"[{query_type}]\n{user_input}"
        if image_data:
            full_prompt += "\nAlso analyze the uploaded image."

        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            # Send message via Gemini GenerativeModel
            response = st.session_state.chat.send_message(full_prompt)
            st.session_state.chat_history.append({"role": "assistant", "message": response.text})
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"Error communicating with LLM: {e}")

    # Display previous chat history
    if st.session_state.chat_history:
        display_chat_history(st.session_state.chat_history)

else:
    st.info("Please enter your Gemini API key in the sidebar to start chatting.")
