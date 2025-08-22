import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    .input-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-high-risk {
        background: #fef2f2;
        border: 2px solid #dc2626;
        color: #991b1b;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-low-risk {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        color: #14532d;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .info-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 12px 24px;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        font-size: 1rem;
        cursor: pointer;
        width: 100%;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 6px;
        width: 100%;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .metric-label {
        font-weight: 500;
        color: #374151;
    }
    
    .metric-value {
        color: #6b7280;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .results-container {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 8px;
        background: #fafbfc;
        border: 1px solid #e1e5e9;
    }
    
    .scroll-target {
        scroll-margin-top: 20px;
    }
</style>

<script>
function scrollToResults() {
    setTimeout(function() {
        const resultsElement = document.querySelector('.results-container');
        if (resultsElement) {
            resultsElement.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    }, 100);
}
</script>
""", unsafe_allow_html=True)

# Load saved model, scaler, and expected columns
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("knn_heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        expected_columns = joblib.load("heart_columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None

model, scaler, expected_columns = load_model_components()

# Header
st.markdown('<h1 class="main-header">Heart Disease Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced cardiovascular risk prediction system developed by Satyam</p>', unsafe_allow_html=True)

if model is None:
    st.error("Unable to load the prediction model. Please ensure all model files are available.")
    st.stop()

# Initialize session state for prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None

# Main content layout
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "About", "Guidelines"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Demographics Section
        st.markdown('<div class="section-header">Patient Demographics</div>', unsafe_allow_html=True)
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            age = st.slider("Age (years)", 18, 100, 40)
        with demo_col2:
            sex = st.selectbox("Sex", ["M", "F"], index=0)
        
        # Vital Signs Section
        st.markdown('<div class="section-header">Vital Signs & Laboratory Results</div>', unsafe_allow_html=True)
        
        vital_col1, vital_col2 = st.columns(2)
        with vital_col1:
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 
                                       min_value=80, max_value=200, value=120)
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 
                                        min_value=100, max_value=600, value=200)
            max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        
        with vital_col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", 
                                    options=[0, 1], 
                                    format_func=lambda x: "Yes" if x == 1 else "No")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        
        # Clinical Symptoms Section
        st.markdown('<div class="section-header">Clinical Symptoms & Exercise Response</div>', unsafe_allow_html=True)
        
        symptom_col1, symptom_col2 = st.columns(2)
        with symptom_col1:
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        with symptom_col2:
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    with col2:
        st.markdown('<div class="section-header">Current Assessment</div>', unsafe_allow_html=True)
        
        # Patient summary
        st.markdown("**Patient Summary**")
        
        summary_items = [
            ("Age", f"{age} years"),
            ("Sex", sex),
            ("Blood Pressure", f"{resting_bp} mmHg"),
            ("Cholesterol", f"{cholesterol} mg/dL"),
            ("Max Heart Rate", f"{max_hr} bpm"),
            ("Chest Pain", chest_pain),
        ]
        
        for label, value in summary_items:
            st.markdown(f'<div class="metric-row"><span class="metric-label">{label}:</span><span class="metric-value">{value}</span></div>', unsafe_allow_html=True)
    
    # Prediction button - moved outside columns for better placement
    st.markdown("---")
    
    # Center the button
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        if st.button("Generate Risk Assessment", type="primary", use_container_width=True):
            with st.spinner("Processing assessment..."):
                # Create input dictionary
                raw_input = {
                    'Age': age,
                    'RestingBP': resting_bp,
                    'Cholesterol': cholesterol,
                    'FastingBS': fasting_bs,
                    'MaxHR': max_hr,
                    'Oldpeak': oldpeak,
                    'Sex_' + sex: 1,
                    'ChestPainType_' + chest_pain: 1,
                    'RestingECG_' + resting_ecg: 1,
                    'ExerciseAngina_' + exercise_angina: 1,
                    'ST_Slope_' + st_slope: 1
                }

                # Create and process dataframe
                input_df = pd.DataFrame([raw_input])
                
                for col in expected_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[expected_columns]
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]
                
                # Get confidence if available
                try:
                    prediction_proba = model.predict_proba(scaled_input)[0]
                    confidence = max(prediction_proba) * 100
                except:
                    confidence = None
                
                # Store results in session state
                st.session_state.prediction_made = True
                st.session_state.prediction_result = prediction
                st.session_state.confidence_score = confidence
                
                # Add a small delay for smooth transition
                time.sleep(0.5)

    # Display results immediately below the button
    if st.session_state.prediction_made:
        st.markdown('<div class="results-container scroll-target">', unsafe_allow_html=True)
        
        # Add JavaScript for auto-scroll
        st.markdown("""
        <script>
        setTimeout(function() {
            const resultsContainer = document.querySelector('.results-container');
            if (resultsContainer) {
                resultsContainer.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center' 
                });
            }
        }, 200);
        </script>
        """, unsafe_allow_html=True)
        
        st.markdown("### Assessment Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            if st.session_state.prediction_result == 1:
                st.markdown(f'''
                <div class="result-high-risk">
                    HIGH RISK
                    <br><small>Risk of heart disease detected</small>
                    {f"<br><small>Confidence: {st.session_state.confidence_score:.1f}%</small>" if st.session_state.confidence_score else ""}
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("**Recommended Actions:**")
                st.markdown("""
                - Consult with a cardiologist immediately
                - Complete comprehensive cardiac evaluation
                - Discuss treatment options
                - Monitor cardiovascular symptoms closely
                - Consider immediate lifestyle modifications
                - Schedule follow-up appointments
                """)
                
            else:
                st.markdown(f'''
                <div class="result-low-risk">
                    LOW RISK
                    <br><small>No significant risk detected</small>
                    {f"<br><small>Confidence: {st.session_state.confidence_score:.1f}%</small>" if st.session_state.confidence_score else ""}
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown("**Recommended Actions:**")
                st.markdown("""
                - Continue current health practices
                - Maintain regular exercise routine
                - Schedule routine health check-ups
                - Monitor blood pressure and cholesterol
                - Stay informed about heart health
                - Maintain a heart-healthy diet
                """)
        
        with result_col2:
            # Confidence gauge
            if st.session_state.confidence_score:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = st.session_state.confidence_score,
                    title = {'text': "Assessment Confidence", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#dc2626" if st.session_state.prediction_result == 1 else "#16a34a"},
                        'steps': [
                            {'range': [0, 70], 'color': "#f3f4f6"},
                            {'range': [70, 90], 'color': "#e5e7eb"}
                        ],
                        'threshold': {
                            'line': {'color': "#374151", 'width': 3},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig.update_layout(
                    height=300, 
                    font={'size': 12},
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional info card
           
        
        # Reset button
        if st.button("Reset Assessment", type="secondary", use_container_width=True):
            st.session_state.prediction_made = False
            st.session_state.prediction_result = None
            st.session_state.confidence_score = None
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### This system utilizes machine learning algorithms to analyze clinical parameters and provide cardiovascular risk assessment. Built for healthcare professionals and researchers to support clinical decision-making through evidence-based risk stratification For educational and screening purposes only. Not a substitute for professional medical advice.")
    
   
with tab3:
    st.markdown("### Clinical Guidelines & Reference Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Blood Pressure Categories:**
        - Normal: < 120/80 mmHg
        - Elevated: 120-129/<80 mmHg
        - Stage 1: 130-139/80-89 mmHg
        - Stage 2: ≥140/90 mmHg
        
        **Cholesterol Levels:**
        - Desirable: < 200 mg/dL
        - Borderline: 200-239 mg/dL
        - High: ≥ 240 mg/dL
        """)
    
    with col2:
        st.markdown("""
        **ECG Classifications:**
        - Normal: No significant abnormalities
        - ST: ST-T wave abnormality
        - LVH: Left ventricular hypertrophy
        
        **Chest Pain Types:**
        - TA: Typical Angina
        - ATA: Atypical Angina  
        - NAP: Non-Anginal Pain
        - ASY: Asymptomatic
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem; font-size: 0.9rem;">
<p>Heart Disease Risk Assessment System | Developed by Satyam</p>
<p>For educational purposes only - Not a substitute for professional medical advice</p>
</div>
""", unsafe_allow_html=True)