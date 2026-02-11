import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Fetal Down Syndrome Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# SESSION STATE
# ----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------------------
# MODEL PATHS (FIXED)
# ----------------------------
MODEL_PATHS = {
    "Optimized Model": "final_model_optimized.h5",
    "Best Optimized": "best_model_optimized.h5",
    "Simplified Model": "final_model_simplified.h5",
    "Best Simplified": "best_model_simplified.h5",
    "Best Model": "best_down_syndrome_model.h5"
}

# ----------------------------
# MODEL LOADER (FIXED)
# ----------------------------
@st.cache_resource
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
def preprocess_image(image, target_size=(224, 224)):

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ----------------------------
# GAUGE CHART
# ----------------------------
def create_gauge_chart(prob):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': '#27ae60'},
                {'range': [30, 70], 'color': '#f39c12'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ]
        }
    ))

    fig.update_layout(height=280)
    return fig

# ----------------------------
# MAIN APP
# ----------------------------
def main():

    st.title("🏥 Fetal Down Syndrome Detection System")

    # Sidebar
    with st.sidebar:

        selected_model = st.selectbox(
            "Select Model",
            list(MODEL_PATHS.keys())
        )

        threshold = st.slider(
            "Decision Threshold",
            0.0, 1.0, 0.5
        )

    # Upload
    uploaded = st.file_uploader(
        "Upload ultrasound image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):

            model = load_model(MODEL_PATHS[selected_model])

            if model is None:
                return

            img = preprocess_image(image)

            prediction = model.predict(img, verbose=0)
            probability = float(prediction[0][0])

            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

            if probability >= threshold:
                st.error(f"⚠️ HIGH RISK ({probability*100:.2f}%)")
            else:
                st.success(f"✅ LOW RISK ({(1-probability)*100:.2f}%)")

            # Save history
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "file": uploaded.name,
                "prob": probability
            })

    # History
    if st.session_state.history:

        st.subheader("History")

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
