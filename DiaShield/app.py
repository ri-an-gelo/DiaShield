import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ----------------- Configuration & Setup -----------------
st.set_page_config(page_title="DiaShield | Predictive Analytics", layout="wide", initial_sidebar_state="expanded")

# Define High-End Professional Palette
COLOR_LOW_RISK = "#34d399"   # Bright Emerald
COLOR_MODERATE_RISK = "#fbbf24" # Bright Amber
COLOR_HIGH_RISK = "#f87171"  # Bright Coral Red

# ----------------- Premium Glassmorphism & Animations -----------------
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* Global Font styling */
        html, body {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Ambient glowing background for the main app */
        .stApp {
            background-color: #0f172a;
            background-image: 
                radial-gradient(at 10% 20%, rgba(37, 99, 235, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* Glassmorphism Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Glassmorphism Cards for Markdown elements */
        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        /* Titles and text styling */
        h1 {
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            letter-spacing: -1px;
            margin-bottom: 0.5rem;
        }
        
        h2, h3 {
            color: #f8fafc;
            font-weight: 600;
            letter-spacing: -0.5px;
        }

        p {
            color: #cbd5e1;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ----------------- Load Resources -----------------
MODEL_PATH = 'models/diabetes_model.pkl'
DATA_PATH = 'data/preprocessed_dataset.csv'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

# ----------------- Initialize Session States -----------------
if "ran_prediction" not in st.session_state:
    st.session_state.ran_prediction = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prob" not in st.session_state:
    st.session_state.prob = 0.0
if "risk_str" not in st.session_state:
    st.session_state.risk_str = ""
if "ui_stats" not in st.session_state:
    st.session_state.ui_stats = {}

# ----------------- UI Header -----------------
st.markdown("<h1>DiaShield</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card" style="border-left: 4px solid #3b82f6;">
    <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
    <strong>Disclaimer:</strong> This application utilizes a Random Forest algorithm to estimate risk probability based on demographic and biometric statistics. It does not replace diagnostics provided by healthcare professionals.
    </p>
</div>
""", unsafe_allow_html=True)

model = load_model()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h3 style='color: #60a5fa;'>Patient Overview</h3>", unsafe_allow_html=True)
    age = st.slider("Age", 1, 120, 30, help="Patient age in years")
    
    st.markdown("<br><h3 style='color: #60a5fa;'>Biometrics</h3>", unsafe_allow_html=True)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (m)", min_value=0.5, max_value=3.0, value=1.75, step=0.01)
    
    calculated_bmi = weight / (height ** 2)
    st.caption(f"Calculated BMI: {calculated_bmi:.1f} kg/m²")
    
    st.markdown("<br><h3 style='color: #60a5fa;'>Medical History</h3>", unsafe_allow_html=True)
    family_history = st.radio("Genetic disposition to Diabetes", ["No", "Yes"])
    
    st.divider()
    st.markdown("<h3 style='color: #60a5fa;'>Lifestyle Indicators</h3>", unsafe_allow_html=True)
    exercise = st.selectbox("Activity Level", ["Low", "Medium", "High"])
    sleep = st.number_input("Sleep Matrix (Hours)", 0, 24, 7)
    fast_food = st.selectbox("Processed Food Frequency", ["Never", "Rarely", "Often", "Very Often"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Risk Analytics"):
        st.session_state.ran_prediction = True
        
        pedigree_val = 1.0 if family_history == "Yes" else 0.2
        
        input_df = pd.DataFrame({
            'Age': [age],
            'BMI': [calculated_bmi],
            'DiabetesPedigreeFunction': [pedigree_val]
        })
        
        if model is not None:
            raw_prob = model.predict_proba(input_df)[0][1] * 100
            
            if raw_prob < 30:
                risk_str = "Low Risk Profile"
                risk_col = COLOR_LOW_RISK
            elif raw_prob < 50:
                risk_str = "Moderate Risk Profile"
                risk_col = COLOR_MODERATE_RISK
            else:
                risk_str = "Elevated Risk Profile"
                risk_col = COLOR_HIGH_RISK
                
            st.session_state.prob = raw_prob
            st.session_state.risk_str = risk_str
            st.session_state.risk_col = risk_col
            
            st.session_state.ui_stats = {
                "BMI": calculated_bmi,
                "Exercise": exercise,
                "Sleep": sleep,
                "FastFood": fast_food,
                "Age": age,
                "FamilyHistory": family_history
            }
            
            p = st.session_state.prob
            risk_level = st.session_state.risk_str
            
            init_msg = f"System Processing Complete. The patient falls into the **{risk_level}** ({p:.1f}% deviation).\n\n"
            
            if calculated_bmi >= 25:
                init_msg += "• **Calculated Mass:** BMI falls into overweight thresholds. Targeted metabolic exercise is recommended.\n"
            if exercise == "Low":
                init_msg += "• **Kinetic Output:** Elevating resting heart rate for 30 minutes daily will rapidly enhance insulin sensitivity.\n"
            if sleep < 7:
                init_msg += "• **Recovery Deficit:** Cortisol levels rise when sleep drops below 7 hours. A disciplined circadian rhythm is advised.\n"
            if fast_food in ["Often", "Very Often"]:
                init_msg += "• **Nutritional Profile:** Reduce processed synthesis. Implementing a complex-carbohydrate based diet lowers volatility.\n"
            if risk_level == "Low Risk Profile" and calculated_bmi < 25 and exercise != "Low":
                init_msg += "• **Optimization Reached:** The physiological parameters are stable. No immediate disruptions are recommended."
                
            st.session_state.messages = [{"role": "assistant", "content": init_msg}]

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["Analysis Insight", "AI Diagnostic Agent", "Dataset Analytics"])

if st.session_state.ran_prediction:
    with tab1:
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid {st.session_state.risk_col}; height: 100%;">
                <p style="text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem; margin:0;">Categorical Indicator</p>
                <h2 style='color: {st.session_state.risk_col}; margin-top: 0.2rem; font-size: 2.2rem;'>{st.session_state.risk_str}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid #60a5fa; height: 100%;">
                <p style="text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem; margin:0;">Statistical Probability</p>
                <h2 style='color: #f8fafc; margin-top: 0.2rem; font-size: 2.2rem;'>{st.session_state.prob:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><h3>Diagnostic Adjustments</h3>", unsafe_allow_html=True)
        
        bmi_stat = st.session_state.ui_stats.get('BMI', 0)
        ex_stat = st.session_state.ui_stats.get('Exercise', "Low")
        slp_stat = st.session_state.ui_stats.get('Sleep', 0)
        ff_stat = st.session_state.ui_stats.get('FastFood', "Often")
        
        # Native alerts conform nicely to the dark theme via config.toml
        if bmi_stat >= 25:
            st.warning("Weight Management Protocol: Initiate structured calorie deficit interventions immediately to stabilize BMI.")
        else:
            st.success("Target Mass Achieved: Current BMI falls cleanly within standard functional parameters.")
            
        if ex_stat == "Low":
            st.error("Hypokinetic Warning: Sedentary lifestyle severely limits cellular insulin processing. Aerobic activity is required.")
            
        if slp_stat < 7:
            st.info("Sleep Continuity Request: Inadequate REM cycles elevate systemic stress. Patient intervention to increase duration to 8 hours is optimal.")
            
        if ff_stat in ["Often", "Very Often"]:
            st.warning("Processed Diet Alert: The influx of simple sugars and saturated lipids directly counters metabolic homeostasis.")

    with tab2:
        st.markdown("""
        <div class="glass-card">
            <p style="margin: 0; font-size:0.95rem;">You are communicating with the local diagnostics simulator. Data is not transmitted externally.</p>
        </div>
        """, unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Enter clinical query..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            mock_reply = ""
            p_lower = prompt.lower()
            if "risk" in p_lower:
                mock_reply = f"The derived score holds the subject at a **{st.session_state.risk_str}**. Age, cellular resistance indicators (BMI), and strict pedigree records formed this determination."
            elif "improve" in p_lower or "intervention" in p_lower:
                mock_reply = "Direct modifications to energy burn (via medium-high exercise) coupled with dropping carbohydrate-dense fast food will aggressively curve the statistical probability downward."
            elif "explain" in p_lower or "detail" in p_lower:
                mock_reply = f"The matrix cross-referenced the inputted Age ({st.session_state.ui_stats.get('Age')}), BMI ratio ({st.session_state.ui_stats.get('BMI',0):.1f}), and historical lineage parameters."
            else:
                mock_reply = "Noted. This is an offline, standalone simulation structure focused entirely on predicting Type 2 constraints over the generated baseline."
                
            with st.chat_message("assistant"):
                st.markdown(mock_reply)
            st.session_state.messages.append({"role": "assistant", "content": mock_reply})

else:
    with tab1:
        st.info("Enter subject metrics and initiate 'Generate Risk Analytics' to proceed.")
    with tab2:
        st.info("Secure chat interface locks until primary profile data is rendered.")

with tab3:
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <h3 style="margin-top:0;">Dataset Topography</h3>
        <p style="margin: 0; font-size: 0.95rem;">A high-level inspection of the internal variables used to train the Random Forest predictor. Note how Age and internal BMI heavily segment the populations into dense risk pools.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    # Set global plotting style to dark to match the app
    plt.style.use('dark_background')
    
    if df is not None and not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Demographic Concentration**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            # Set transparent background for matplotlib
            fig1.patch.set_alpha(0.0)
            ax1.patch.set_alpha(0.0)
            
            sns.histplot(data=df, x='Age', hue='Outcome', multiple='stack', palette={0: COLOR_LOW_RISK, 1: COLOR_HIGH_RISK}, ax=ax1)
            ax1.set_ylabel("Frequency", color='#f8fafc')
            ax1.set_xlabel("Age", color='#f8fafc')
            ax1.tick_params(colors='#cbd5e1')
            for spine in ax1.spines.values():
                spine.set_edgecolor('None')
            st.pyplot(fig1)

            if model is not None:
                st.markdown("**Algorithmic Weighting**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                fig2.patch.set_alpha(0.0)
                ax2.patch.set_alpha(0.0)
                
                feats = ['Age', 'BMI', 'Lineage']
                imps = model.feature_importances_
                ax2.barh(feats, imps, color="#3b82f6")
                ax2.set_xlabel("Weight Coefficient", color='#f8fafc')
                ax2.tick_params(colors='#cbd5e1')
                for spine in ax2.spines.values():
                    spine.set_edgecolor('None')
                st.pyplot(fig2)
                
        with c2:
            st.markdown("**BMI Index Distribution**")
            df['BMI_Cat'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            fig3.patch.set_alpha(0.0)
            ax3.patch.set_alpha(0.0)
            
            sns.countplot(data=df, x='BMI_Cat', hue='Outcome', palette={0: COLOR_LOW_RISK, 1: COLOR_HIGH_RISK}, ax=ax3)
            ax3.set_ylabel("Volume", color='#f8fafc')
            ax3.set_xlabel("BMI Category", color='#f8fafc')
            ax3.tick_params(colors='#cbd5e1')
            for spine in ax3.spines.values():
                spine.set_edgecolor('None')
            st.pyplot(fig3)
    else:
        st.warning("System error: Missing dataset topography. Required: model_training.py initialization.")
