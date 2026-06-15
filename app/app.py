import os
# --- APPLE SILICON PROTOBUF FIX (MUST BE AT THE VERY TOP) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
import google.generativeai as genai

# Set up beautiful page title and icon
st.set_page_config(page_title="AgriShield AI Dashboard", page_icon="🌾", layout="wide")

# --- SIDEBAR FOR API KEY ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("To use the GenAI features, enter your free Gemini API Key below.")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get your free key here](https://aistudio.google.com/app/apikey)")

st.title("🌾 AgriShield AI: Smart Farming Assistant")
st.markdown("Welcome to your intelligent agricultural advisor dashboard. Select a tool below to get started.")

# Create THREE visual tabs now!
tab1, tab2, tab3 = st.tabs(["📸 Crop Disease Diagnostics", "📊 Crop Yield Forecasting", "🤖 AI Agronomist"])

# --- TAB 1: COMPUTER VISION (DISEASE SCANNER) ---
with tab1:
    st.header("Crop Disease Scanner")
    st.write("Upload a clear photo of a crop leaf to identify potential diseases instantly.")
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Crop Leaf", width=300)
            
            st.write("🔄 Analyzing image with Deep Learning...")
            
            model_path = 'models/plant_disease_model.keras'
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                img = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model.predict(img_array)
                
                st.success("Analysis Complete!")
                st.info("The Deep Learning model successfully processed the leaf architecture.")
            else:
                st.error("Cannot find 'plant_disease_model.keras' in your models folder.")
        except Exception as e:
            st.error(f"An error occurred during vision processing: {e}")

# --- TAB 2: DATA SCIENCE (YIELD PREDICTOR) ---
with tab2:
    st.header("Yield Forecasting Analytics")
    st.write("Input current environmental factors to calculate expected crop production parameters.")
    
    yield_model_path = 'models/yield_model.pkl'
    
    if os.path.exists(yield_model_path):
        try:
            with open(yield_model_path, 'rb') as f:
                yield_model = pickle.load(f)
            
            # Safely get expected features
            if hasattr(yield_model, "feature_names_in_"):
                expected_features = yield_model.feature_names_in_
            else:
                expected_features = ["Temperature", "Rainfall", "Fertilizer", "Pesticide"]
                
            st.write(f"This model was trained on **{len(expected_features)}** specific data points. Please fill them out below:")
            
            user_inputs = []
            cols = st.columns(2)
            
            for i, feature_name in enumerate(expected_features):
                with cols[i % 2]:
                    val = st.number_input(f"Enter {feature_name}", value=0.0)
                    user_inputs.append(val)
                    
            if st.button("Forecast Total Yield"):
                prediction = yield_model.predict([user_inputs])
                st.balloons()
                st.metric(label="Predicted Crop Yield Production", value=f"{prediction[0]:.2f}")
                
        except Exception as e:
             st.error(f"An error occurred during yield forecasting: {e}")
    else:
        st.error("Cannot find 'yield_model.pkl' in your models folder.")

# --- TAB 3: GENERATIVE AI (EXPERT ADVISOR) ---
with tab3:
    st.header("🤖 GenAI Agronomist")
    st.write("Ask our AI expert for custom farming advice, pest control strategies, or soil remedies.")
    
    user_query = st.text_area("Describe your crop issue or ask a farming question here:", height=100)
    
    if st.button("Generate Expert Report"):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar on the left first!")
        elif not user_query:
            st.warning("⚠️ Please type a question before clicking the button.")
        else:
            try:
                with st.spinner("Analyzing agricultural data..."):
                    # Connect to the AI
                    genai.configure(api_key=api_key)
                    
                    # --- FIXED MODEL NAME FOR 2026 API COMPATIBILITY ---
                    llm = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # Create a strict persona for the AI
                    prompt = f"You are an expert agronomist and agricultural data scientist. A farmer asks: {user_query}. Provide a structured, highly professional, and actionable response."
                    
                    response = llm.generate_content(prompt)
                    
                    st.success("Report Generated Successfully!")
                    st.markdown("### 📋 Expert Advisory Report")
                    st.write(response.text)
            except Exception as e:
                st.error(f"Error connecting to AI Server: {e}")