import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="Fetal Down Syndrome Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --success-color: #27ae60;
        --warning-color: #f39c12;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    
    .danger-card {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Results container */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Model paths - UPDATE THESE WITH YOUR ACTUAL PATHS
MODEL_PATHS = {
    "Optimized Model": r"C:\Users\JAYA SOORYA\Downloads\working_dir_backup (4)\final_model_optimized.h5",
    "Best Optimized": r"C:\Users\JAYA SOORYA\Downloads\working_dir_backup (4)\best_model_optimized.h5",
    "Simplified Model": r"C:\Users\JAYA SOORYA\Downloads\working_dir_backup (4)\final_model_simplified.h5",
    "Best Simplified": r"C:\Users\JAYA SOORYA\Downloads\working_dir_backup (4)\best_model_simplified.h5",
    "Best Model": r"C:\Users\JAYA SOORYA\Downloads\working_dir_backup (4)\best_down_syndrome_model.h5"
}

@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#27ae60'},
                {'range': [30, 70], 'color': '#f39c12'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_confidence_bars(standard_prob, nonstandard_prob):
    """Create horizontal bar chart for class probabilities"""
    fig = go.Figure()
    
    categories = ['Standard', 'Non-standard']
    probabilities = [standard_prob * 100, nonstandard_prob * 100]
    colors = ['#27ae60', '#e74c3c']
    
    fig.add_trace(go.Bar(
        y=categories,
        x=probabilities,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.2f}%' for p in probabilities],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Classification Confidence",
        xaxis_title="Probability (%)",
        yaxis_title="Category",
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def interpret_results(probability, threshold=0.5):
    """Interpret the prediction results"""
    if probability >= threshold:
        risk_level = "HIGH RISK"
        color = "#e74c3c"
        recommendation = """
        ⚠️ **HIGH RISK DETECTED**
        
        The model indicates a higher probability of Down Syndrome markers. 
        
        **Immediate Actions Required:**
        - Consult with a genetic counselor immediately
        - Schedule comprehensive genetic testing (amniocentesis or CVS)
        - Arrange detailed anatomical ultrasound scan
        - Consider additional screening tests (NIPT, triple/quad screen)
        """
    else:
        risk_level = "LOW RISK"
        color = "#27ae60"
        recommendation = """
        ✅ **LOW RISK DETECTED**
        
        The model indicates a lower probability of Down Syndrome markers.
        
        **Recommended Actions:**
        - Continue routine prenatal care
        - Follow standard monitoring protocols
        - Maintain scheduled ultrasound appointments
        - Discuss results with healthcare provider
        """
    
    return risk_level, color, recommendation

# Main App Layout
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Fetal Down Syndrome Detection System</h1>
        <p>AI-Powered Ultrasound Analysis for Prenatal Screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(MODEL_PATHS.keys()),
            help="Choose the trained model for analysis"
        )
        
        # Threshold adjustment
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust the classification threshold"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Information")
        st.info("""
        **Architecture:** MobileNetV2 + Dense Layers
        
        **Input Size:** 224×224×3
        
        **Training Data:** 1,372 ultrasound images
        
        **Anatomical Markers:** 9 key structures
        """)
        
        st.markdown("---")
        
        # Clinical disclaimer
        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This tool is for screening purposes only. 
        All results must be confirmed by qualified 
        healthcare professionals and appropriate 
        diagnostic tests.
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Batch Processing", "📈 History", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Ultrasound Image")
            
            uploaded_file = st.file_uploader(
                "Choose an ultrasound image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a fetal ultrasound image for analysis"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Ultrasound", use_container_width=True)
            
                st.markdown("""
                <style>
                .info-card {
                    color: black;
                }
                </style>
                """, unsafe_allow_html=True)
            
                st.markdown(f"""
                <div class="info-card">
                    <strong>Image Information:</strong><br>
                    📏 Size: {image.size[0]} × {image.size[1]} pixels<br>
                    🎨 Mode: {image.mode}<br>
                    📁 Format: {image.format}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🤖 Analysis Results")
            
            if uploaded_file is not None:
                if st.button("🔬 Analyze Image", type="primary"):
                    with st.spinner("Loading model and analyzing image..."):
                        # Load model
                        model = load_model(MODEL_PATHS[selected_model])
                        
                        if model is not None:
                            # Preprocess image
                            processed_img, display_img = preprocess_image(image)
                            
                            if processed_img is not None:
                                # Make prediction
                                prediction = model.predict(processed_img, verbose=0)
                                probability = float(prediction[0][0])
                                
                                # Store in history
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'filename': uploaded_file.name,
                                    'probability': probability,
                                    'model': selected_model
                                })
                                
                                # Interpret results
                                risk_level, color, recommendation = interpret_results(probability, threshold)
                                
                                # Display results
                                st.markdown(f"""
                                <div style="background-color: {color}; padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
                                    <h2 style="margin: 0;">{risk_level}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Gauge chart
                                st.plotly_chart(
                                    create_gauge_chart(probability, "Non-standard Case Probability"),
                                    use_container_width=True
                                )
                                
                                # Probability bars
                                standard_prob = 1 - probability
                                st.plotly_chart(
                                    create_confidence_bars(standard_prob, probability),
                                    use_container_width=True
                                )
                                
                                # Detailed metrics
                                st.markdown("### 📊 Detailed Metrics")
                                
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                
                                with metrics_col1:
                                    st.metric(
                                        "Standard Probability",
                                        f"{standard_prob*100:.2f}%",
                                        delta=f"{(0.5-standard_prob)*100:.1f}%" if standard_prob < 0.5 else f"+{(standard_prob-0.5)*100:.1f}%"
                                    )
                                
                                with metrics_col2:
                                    st.metric(
                                        "Non-standard Probability",
                                        f"{probability*100:.2f}%",
                                        delta=f"{(probability-0.5)*100:.1f}%" if probability > 0.5 else f"{(0.5-probability)*100:.1f}%"
                                    )
                                
                                with metrics_col3:
                                    confidence = max(standard_prob, probability)
                                    st.metric(
                                        "Confidence Level",
                                        f"{confidence*100:.2f}%"
                                    )
                                
                                # Clinical recommendation
                                st.markdown("### 🩺 Clinical Recommendation")
                                st.markdown(recommendation)
                                
                                # Key markers
                                st.markdown("### 🔬 Anatomical Markers Analyzed")
                                markers_col1, markers_col2, markers_col3 = st.columns(3)
                                
                                with markers_col1:
                                    st.markdown("""
                                    - ✅ NT (Nuchal Translucency)
                                    - ✅ Nasal Bone
                                    - ✅ IT (Intracranial Translucency)
                                    """)
                                
                                with markers_col2:
                                    st.markdown("""
                                    - ✅ CM (Cisterna Magna)
                                    - ✅ Thalami
                                    - ✅ Midbrain
                                    """)
                                
                                with markers_col3:
                                    st.markdown("""
                                    - ✅ Palate
                                    - ✅ Nasal Tip
                                    - ✅ Nasal Skin
                                    """)
                                
                                # Download report
                                st.markdown("### 📄 Generate Report")
                                
                                report_data = {
                                    "Analysis Report": "Fetal Down Syndrome Screening",
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Image": uploaded_file.name,
                                    "Model Used": selected_model,
                                    "Risk Level": risk_level,
                                    "Non-standard Probability": f"{probability*100:.2f}%",
                                    "Standard Probability": f"{standard_prob*100:.2f}%",
                                    "Confidence": f"{confidence*100:.2f}%",
                                    "Threshold Used": f"{threshold*100:.0f}%"
                                }
                                
                                report_df = pd.DataFrame([report_data]).T
                                report_df.columns = ['Value']
                                
                                csv = report_df.to_csv()
                                st.download_button(
                                    label="📥 Download Report (CSV)",
                                    data=csv,
                                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
    
    with tab2:
        st.markdown("### 📊 Batch Processing")
        st.info("Upload multiple ultrasound images for batch analysis")
        
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if batch_files:
            st.write(f"📁 {len(batch_files)} images uploaded")
            
            if st.button("🚀 Process Batch", type="primary"):
                model = load_model(MODEL_PATHS[selected_model])
                
                if model is not None:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(batch_files):
                        image = Image.open(file)
                        processed_img, _ = preprocess_image(image)
                        
                        if processed_img is not None:
                            prediction = model.predict(processed_img, verbose=0)
                            probability = float(prediction[0][0])
                            
                            risk_level, _, _ = interpret_results(probability, threshold)
                            
                            results.append({
                                'Filename': file.name,
                                'Non-standard Probability': f"{probability*100:.2f}%",
                                'Standard Probability': f"{(1-probability)*100:.2f}%",
                                'Risk Level': risk_level,
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        
                        progress_bar.progress((idx + 1) / len(batch_files))
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    high_risk = sum(1 for r in results if r['Risk Level'] == 'HIGH RISK')
                    low_risk = len(results) - high_risk
                    
                    with col1:
                        st.metric("Total Processed", len(results))
                    with col2:
                        st.metric("High Risk Cases", high_risk)
                    with col3:
                        st.metric("Low Risk Cases", low_risk)
                    
                    # Download batch results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Batch Results",
                        csv,
                        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
    
    with tab3:
        st.markdown("### 📈 Analysis History")
        
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Display history table
            st.dataframe(history_df, use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                history_df,
                x='timestamp',
                y='probability',
                color='model',
                hover_data=['filename'],
                title="Prediction History Over Time"
            )
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No analysis history yet. Start analyzing images to build your history.")
    
    with tab4:
        st.markdown("### ℹ️ About This System")
        
        st.markdown("""
        ## 🏥 Fetal Down Syndrome Detection System
        
        This AI-powered system analyzes fetal ultrasound images to assess the risk of Down Syndrome 
        based on multiple anatomical markers.
        
        ### 🔬 Technology
        
        - **Deep Learning Framework:** TensorFlow/Keras
        - **Base Architecture:** MobileNetV2 with transfer learning
        - **Input Resolution:** 224×224 pixels
        - **Training Dataset:** 1,372 annotated ultrasound images
        - **Anatomical Markers:** 9 key fetal structures
        
        ### 📊 Performance Metrics
        
        - **Sensitivity:** 96.02% (Detection rate for Down Syndrome cases)
        - **Specificity:** 48.53% (Correct identification of normal cases)
        - **AUC-ROC:** 0.8577
        - **F1-Score:** 0.8145
        
        ### 🎯 Key Features Analyzed
        
        1. **NT (Nuchal Translucency)** - Primary first-trimester marker
        2. **Nasal Bone** - Presence/absence critical for diagnosis
        3. **IT (Intracranial Translucency)** - Brain development marker
        4. **CM (Cisterna Magna)** - Posterior fossa measurement
        5. **Thalami** - Brain structure assessment
        6. **Midbrain** - Neural development indicator
        7. **Palate** - Facial profile marker
        8. **Nasal Tip & Skin** - Facial feature measurements
        
        ### ⚠️ Important Medical Disclaimer
        
        **This system is designed as a SCREENING TOOL ONLY and should NOT be used as the sole basis 
        for clinical decisions.**
        
        - Results must be confirmed through comprehensive genetic testing
        - All positive screenings require genetic counseling
        - Diagnostic confirmation via amniocentesis or CVS is essential
        - Regular prenatal care and professional medical consultation are mandatory
        
        ### 🔒 Data Privacy
        
        - No images are stored on servers
        - All processing is done in real-time
        - Session data is cleared upon closing the application
        
        ### 👨‍💻 Technical Support
        
        For technical issues or questions about the system, please contact your IT support team.
        
        ### 📚 References
        
        This system was developed based on research in prenatal ultrasound screening and deep 
        learning applications in medical imaging.
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** December 2024  
        **Developed with:** Streamlit, TensorFlow, OpenCV
        """)

if __name__ == "__main__":
    main()
