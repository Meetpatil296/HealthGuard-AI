import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import plotly.graph_objects as go
from utils.llm_explainer import generate_explanation
from utils.db import insert_prediction, get_user_history
import uuid


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🫀",
    layout="wide"
)


# =========================================================
# PREMIUM HEALTHGUARD THEME
# =========================================================

st.markdown("""
<style>

    .stApp {
        background: #0B0F14;
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #10161F;
        border-right: 1px solid #263241;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    p, label, .stMarkdown {
        color: #CBD5E1;
    }

    div[data-testid="stMetric"] {
        background: #151B23;
        border: 1px solid #263241;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00B8D9, #4F8CFF);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 184, 217, 0.25);
    }

    .streamlit-expanderHeader {
        background: #151B23;
        border-radius: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #263241;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    models_dir = os.path.join(
        os.path.dirname(__file__),
        "models"
    )

    try:

        model = joblib.load(
            os.path.join(
                models_dir,
                "heart_disease_model.joblib"
            )
        )

        preprocessor = joblib.load(
            os.path.join(
                models_dir,
                "preprocessor.joblib"
            )
        )

        feature_names = joblib.load(
            os.path.join(
                models_dir,
                "feature_names.joblib"
            )
        )

        explainer = shap.TreeExplainer(model)

        return (
            model,
            preprocessor,
            explainer,
            feature_names
        )

    except Exception as e:

        st.error(
            f"Error loading models: {e}"
        )

        return (
            None,
            None,
            None,
            None
        )


model, preprocessor, explainer, feature_names = load_models()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "# 🫀 HealthGuard AI"
)

st.markdown(
    "### Intelligent Heart Disease Risk Assessment & AI Health Assistant"
)

st.caption(
    "Machine Learning • Explainable AI • Personalized Health Insights"
)

st.divider()

# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It is driven by an AI model and is NOT a substitute for "
    "professional medical advice, diagnosis, or treatment."
)


# =========================================================
# MODEL CHECK
# =========================================================

if model is None:

    st.error(
        "Models not found. Please run the training scripts first."
    )

    st.stop()


# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([1, 2])


# =========================================================
# PATIENT DATA INPUT
# =========================================================

with col1:

    st.header("Patient Data Input")

    st.markdown(
        "Please fill out your profile. Clinical tests can be "
        "left as 'Unknown' if you don't have the data."
    )


    # -----------------------------------------------------
    # GENERAL PROFILE
    # -----------------------------------------------------

    with st.expander(
        "1. General Profile",
        expanded=True
    ):

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            help="Your chronological age in years."
        )

        sex = st.selectbox(
            "Sex",
            options=[
                "Male",
                "Female"
            ]
        )


    # -----------------------------------------------------
    # LIFESTYLE
    # -----------------------------------------------------

    with st.expander(
        "2. Lifestyle & Habits",
        expanded=True
    ):

        smoking = st.selectbox(
            "Do you smoke?",
            options=[
                "No",
                "Yes"
            ]
        )

        cigarettes_per_day = 0

        if smoking == "Yes":

            cigarettes_per_day = st.number_input(
                "How many cigarettes a day?",
                min_value=1,
                value=10
            )

        alcohol = st.selectbox(
            "Do you consume alcohol?",
            options=[
                "No",
                "Yes"
            ]
        )

        alcohol_freq = "None"

        if alcohol == "Yes":

            alcohol_freq = st.selectbox(
                "How often do you consume alcohol?",
                options=[
                    "Occasionally",
                    "Weekly",
                    "Daily"
                ]
            )


    # -----------------------------------------------------
    # BASIC VITALS
    # -----------------------------------------------------

    with st.expander(
        "3. Basic Vitals & History",
        expanded=True
    ):

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120,
            help="Normal is around 120/80."
        )

        high_blood_pressure = st.selectbox(
            "Diagnosed with High Blood Pressure?",
            options=[
                "Unknown",
                "No",
                "Yes"
            ]
        )

        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=50,
            max_value=250,
            value=150
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[
                "No",
                "Yes"
            ]
        )

        diabetes = st.selectbox(
            "Diagnosed with Diabetes?",
            options=[
                "Unknown",
                "No",
                "Yes"
            ]
        )


    # -----------------------------------------------------
    # ADVANCED CLINICAL TESTS
    # -----------------------------------------------------

    with st.expander(
        "4. Advanced Clinical Tests (Optional)",
        expanded=True
    ):

        st.info(
            "Leave fields blank or as 'Unknown' if you do not "
            "have these lab results."
        )


        # Chest pain

        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                "Unknown",
                1,
                2,
                3,
                4
            ],
            format_func=lambda x:
                "Unknown"
                if x == "Unknown"
                else {
                    1: "Typical Angina",
                    2: "Atypical Angina",
                    3: "Non-anginal Pain",
                    4: "Asymptomatic"
                }[x]
        )


        # Cholesterol

        chol_val = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100.0,
            max_value=600.0,
            value=None,
            placeholder="Type value if known..."
        )


        # ECG

        restecg = st.selectbox(
            "Resting Electrocardiographic Results",
            options=[
                "Unknown",
                0,
                1,
                2
            ],
            format_func=lambda x:
                "Unknown"
                if x == "Unknown"
                else {
                    0: "Normal",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x]
        )


        # Exercise angina

        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[
                "Unknown",
                "No",
                "Yes"
            ]
        )


        # Oldpeak

        oldpeak_val = st.number_input(
            "ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=None,
            step=0.1,
            placeholder="Type value if known..."
        )


        # Slope

        slope = st.selectbox(
            "Slope of the Peak Exercise ST Segment",
            options=[
                "Unknown",
                1,
                2,
                3
            ],
            format_func=lambda x:
                "Unknown"
                if x == "Unknown"
                else {
                    1: "Upsloping",
                    2: "Flat",
                    3: "Downsloping"
                }[x]
        )


        # Major vessels

        ca = st.selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            options=[
                "Unknown",
                0,
                1,
                2,
                3
            ]
        )


        # Thalassemia

        thal = st.selectbox(
            "Thalassemia",
            options=[
                "Unknown",
                3,
                6,
                7
            ],
            format_func=lambda x:
                "Unknown"
                if x == "Unknown"
                else {
                    3: "Normal",
                    6: "Fixed Defect",
                    7: "Reversible Defect"
                }[x]
        )


        st.markdown("---")

        st.markdown(
            "**Blood & Heart Failure Lab Results:**"
        )


        # Anaemia

        anaemia = st.selectbox(
            "Anaemia (Decrease of red blood cells)?",
            options=[
                "Unknown",
                "No",
                "Yes"
            ]
        )


        # Platelets

        platelets_val = st.number_input(
            "Platelets (kiloplatelets/mL)",
            min_value=10000.0,
            max_value=900000.0,
            value=None,
            step=1000.0,
            placeholder="Type value if known..."
        )


        # CPK

        cpk_val = st.number_input(
            "Creatinine Phosphokinase (CPK enzyme level) (mcg/L)",
            min_value=10.0,
            max_value=10000.0,
            value=None,
            placeholder="Type value if known..."
        )


        # Ejection fraction

        ef_val = st.number_input(
            "Ejection Fraction (%)",
            min_value=10.0,
            max_value=90.0,
            value=None,
            placeholder="Type value if known..."
        )


        # Serum creatinine

        sc_val = st.number_input(
            "Serum Creatinine (mg/dL)",
            min_value=0.1,
            max_value=10.0,
            value=None,
            step=0.1,
            placeholder="Type value if known..."
        )


        # Serum sodium

        ss_val = st.number_input(
            "Serum Sodium (mEq/L)",
            min_value=100.0,
            max_value=160.0,
            value=None,
            step=1.0,
            placeholder="Type value if known..."
        )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    submit_button = st.button(
        "🔍 Analyze Risk",
        type="primary",
        use_container_width=True
    )


# =========================================================
# RESULTS
# =========================================================

with col2:

    if submit_button:

        # =================================================
        # HELPER FUNCTIONS
        # =================================================

        def parse_opt(value):

            if value == "Unknown" or value is None:
                return np.nan

            return value


        def parse_bin(value):

            if value == "Unknown" or value is None:
                return np.nan

            if value == "Yes":
                return 1

            return 0


        # =================================================
        # INPUT DATA
        # =================================================

        inputs_dict = {

            "age": age,

            "trestbps": trestbps,

            "chol": parse_opt(chol_val),

            "thalach": thalach,

            "oldpeak": parse_opt(oldpeak_val),

            "ca": parse_opt(ca),

            "anaemia": parse_bin(anaemia),

            "creatinine_phosphokinase":
                parse_opt(cpk_val),

            "diabetes":
                parse_bin(diabetes),

            "ejection_fraction":
                parse_opt(ef_val),

            "high_blood_pressure":
                parse_bin(high_blood_pressure),

            "platelets":
                parse_opt(platelets_val),

            "serum_creatinine":
                parse_opt(sc_val),

            "serum_sodium":
                parse_opt(ss_val),

            "smoking":
                1 if smoking == "Yes" else 0,

            "sex":
                1 if sex == "Male" else 0,

            "cp":
                parse_opt(cp),

            "fbs":
                1 if fbs == "Yes" else 0,

            "restecg":
                parse_opt(restecg),

            "exang":
                parse_bin(exang),

            "slope":
                parse_opt(slope),

            "thal":
                parse_opt(thal)
        }


        # =================================================
        # DATAFRAME
        # =================================================

        input_df = pd.DataFrame(
            [inputs_dict]
        )[feature_names]


        # =================================================
        # PREPROCESSING
        # =================================================

        processed_input = preprocessor.transform(
            input_df
        )


        # =================================================
        # PREDICTION
        # =================================================

        risk_score = float(
            model.predict_proba(
                processed_input
            )[0][1]
        )


        # =================================================
        # ANALYSIS HEADER
        # =================================================

        st.header(
            "Analysis Results"
        )


        # =================================================
        # RISK CLASSIFICATION
        # =================================================

        if risk_score < 0.30:

            risk_level = "Low"
            risk_icon = "🟢"
            risk_message = (
                "Your predicted risk is relatively low."
            )

        elif risk_score < 0.70:

            risk_level = "Moderate"
            risk_icon = "🟡"
            risk_message = (
                "Your predicted risk is in the moderate range."
            )

        else:

            risk_level = "High"
            risk_icon = "🔴"
            risk_message = (
                "Your predicted risk is elevated."
            )


        # =================================================
        # DASHBOARD METRICS
        # =================================================

        m1, m2, m3 = st.columns(3)


        with m1:

            st.metric(
                "Risk Score",
                f"{risk_score * 100:.1f}%"
            )


        with m2:

            st.metric(
                "Risk Category",
                f"{risk_icon} {risk_level}"
            )


        with m3:

            st.metric(
                "Model",
                "HGB Classifier"
            )


        st.caption(
            risk_message
        )


        # =================================================
        # RISK GAUGE
        # =================================================

        fig_gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=risk_score * 100,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 50,
                        "color": "#F8FAFC"
                    }
                },

                domain={
                    "x": [0, 1],
                    "y": [0, 1]
                },

                title={
                    "text": "Heart Disease Risk Score",
                    "font": {
                        "size": 24,
                        "color": "#F8FAFC"
                    }
                },

                gauge={

                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#94A3B8"
                    },

                    "bar": {
                        "color": "#00B8D9"
                    },

                    "bgcolor": "#151B23",

                    "borderwidth": 1,

                    "bordercolor": "#263241",

                    "steps": [

                        {
                            "range": [0, 30],
                            "color": "#163A2A"
                        },

                        {
                            "range": [30, 70],
                            "color": "#3A3216"
                        },

                        {
                            "range": [70, 100],
                            "color": "#3A1C22"
                        }
                    ]
                }
            )
        )


        fig_gauge.update_layout(

            height=350,

            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            ),

            paper_bgcolor="#0B0F14",

            plot_bgcolor="#0B0F14",

            font={
                "color": "#F8FAFC"
            }
        )


        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )


        # =================================================
        # RISK MESSAGE
        # =================================================

        if risk_score < 0.30:

            st.success(
                "You are in the **Low Risk** category. "
                "Keep up the good work!"
            )

        elif risk_score < 0.70:

            st.warning(
                "You are in the **Moderate Risk** category. "
                "Consider reviewing your lifestyle habits."
            )

        else:

            st.error(
                "You are in the **High Risk** category. "
                "Please consult a healthcare professional."
            )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        st.subheader(
            "🧠 What drove this prediction?"
        )

        st.markdown(
            "This chart breaks down the most significant "
            "factors in your profile. Factors increasing "
            "risk are shown in red, while factors reducing "
            "risk are shown in green."
        )


        with st.spinner(
            "Analyzing driving factors..."
        ):

            shap_values = explainer.shap_values(
                processed_input
            )


            if isinstance(shap_values, list):

                shap_vals = shap_values[1][0]

            else:

                shap_vals = shap_values[0]


            # -------------------------------------------------
            # HUMAN READABLE FEATURE NAMES
            # -------------------------------------------------

            readable_names = {

                "age": "Age",

                "sex": "Gender",

                "cp": "Chest Pain Type",

                "trestbps":
                    "Resting Blood Pressure",

                "chol":
                    "Cholesterol",

                "fbs":
                    "Fasting Blood Sugar",

                "restecg":
                    "Resting ECG",

                "thalach":
                    "Max Heart Rate",

                "exang":
                    "Exercise Angina",

                "oldpeak":
                    "ST Depression (Exercise)",

                "slope":
                    "ST Segment Slope",

                "ca":
                    "Major Vessels Blocked",

                "thal":
                    "Thalassemia",

                "anaemia":
                    "Anaemia",

                "creatinine_phosphokinase":
                    "CPK Enzyme Level",

                "diabetes":
                    "Diabetes",

                "ejection_fraction":
                    "Ejection Fraction",

                "high_blood_pressure":
                    "High Blood Pressure",

                "platelets":
                    "Platelets Count",

                "serum_creatinine":
                    "Serum Creatinine",

                "serum_sodium":
                    "Serum Sodium",

                "smoking":
                    "Smoking Habit"
            }


            # -------------------------------------------------
            # FEATURE IMPACTS
            # -------------------------------------------------

            feature_impacts = {

                readable_names.get(
                    feature_names[i],
                    feature_names[i]
                ): float(shap_vals[i])

                for i in range(
                    len(feature_names)
                )
            }


            sorted_impacts = sorted(

                feature_impacts.items(),

                key=lambda x: abs(x[1]),

                reverse=True

            )[:6]


            top_factors_dict = {
                key: value
                for key, value in sorted_impacts
            }


            # -------------------------------------------------
            # SHAP BAR CHART
            # -------------------------------------------------

            factors = [
                key
                for key, value in reversed(
                    sorted_impacts
                )
            ]


            impacts = [
                value
                for key, value in reversed(
                    sorted_impacts
                )
            ]


            colors = [

                "#ff4b4b"
                if value > 0
                else "#00cc96"

                for value in impacts
            ]


            fig_bar = go.Figure(

                go.Bar(

                    x=impacts,

                    y=factors,

                    orientation="h",

                    marker_color=colors,

                    text=[

                        (
                            f"+{value:.2f}"
                            if value > 0
                            else f"{value:.2f}"
                        )

                        for value in impacts
                    ],

                    textposition="auto"
                )
            )


            fig_bar.update_layout(

                title={
                    "text": "Top 6 Contributing Factors",
                    "font": {
                        "color": "#F8FAFC"
                    }
                },

                xaxis_title="Impact on Risk Score",

                yaxis_title="",

                height=400,

                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                ),

                paper_bgcolor="#0B0F14",

                plot_bgcolor="#0B0F14",

                font={
                    "color": "#F8FAFC"
                },

                xaxis={

                    "zeroline": True,

                    "zerolinewidth": 2,

                    "zerolinecolor": "#64748B",

                    "gridcolor": "#263241"
                },

                yaxis={
                    "gridcolor": "#263241"
                }
            )


            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )


        # =================================================
        # GROQ AI HEALTH ASSISTANT
        # =================================================

        st.subheader(
            "🤖 AI Health Assistant Explanation"
        )


        with st.spinner(
            "Generating personalized explanation..."
        ):

            original_inputs = inputs_dict.copy()


            original_inputs[
                "cigarettes_per_day"
            ] = (
                cigarettes_per_day
                if smoking == "Yes"
                else 0
            )


            original_inputs[
                "alcohol"
            ] = alcohol


            original_inputs[
                "alcohol_freq"
            ] = (
                alcohol_freq
                if alcohol == "Yes"
                else "None"
            )


            explanation = generate_explanation(

                risk_score,

                top_factors_dict,

                original_inputs
            )


            st.markdown(
                explanation
            )


        # =================================================
        # SAVE PREDICTION TO SUPABASE
        # =================================================

        with st.spinner(
            "Saving to history..."
        ):

            inserted = insert_prediction(

                st.session_state.user_id,

                original_inputs,

                risk_score,

                top_factors_dict,

                explanation
            )


            if not inserted:

                st.caption(
                    "Note: Database not connected "
                    "or failed to save history."
                )


# =========================================================
# PATIENT HISTORY
# =========================================================

st.divider()


with st.expander(
    "📋 View Patient History"
):

    history = get_user_history(
        st.session_state.user_id
    )


    if history:

        for record in history:

            st.markdown(
                f"""
                **Date:** {record['created_at'][:10]}

                **Risk Score:** {record['risk_score']:.1%}

                ---
                """
            )

    else:

        st.write(
            "No previous history found."
        )