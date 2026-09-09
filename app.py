import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import plotly.graph_objects as go
import uuid

from utils.llm_explainer import generate_explanation
from utils.db import insert_prediction, get_user_history


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

st.markdown(
    """
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

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #263241;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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

        # SHAP TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None

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

st.markdown("# 🫀 HealthGuard AI")

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
        "Models not found. Please check your models folder."
    )

    st.stop()


# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns(
    [1, 2],
    gap="large"
)


# =========================================================
# PATIENT INPUT
# =========================================================

with col1:

    st.header("👤 Patient Data Input")

    st.caption(
        "Enter available health information to generate "
        "an AI-based risk assessment."
    )

    # =====================================================
    # GENERAL PROFILE
    # =====================================================

    with st.expander(
        "1. General Profile",
        expanded=True
    ):

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            help="Age in years."
        )

        sex = st.selectbox(
            "Sex",
            [
                "Male",
                "Female"
            ]
        )

    # =====================================================
    # LIFESTYLE
    # =====================================================

    with st.expander(
        "2. Lifestyle & Habits",
        expanded=True
    ):

        smoking = st.selectbox(
            "Do you smoke?",
            [
                "No",
                "Yes"
            ]
        )

        cigarettes_per_day = 0

        if smoking == "Yes":

            cigarettes_per_day = st.number_input(
                "Cigarettes per day",
                min_value=1,
                max_value=100,
                value=10
            )

        alcohol = st.selectbox(
            "Do you consume alcohol?",
            [
                "No",
                "Yes"
            ]
        )

        alcohol_freq = "None"

        if alcohol == "Yes":

            alcohol_freq = st.selectbox(
                "Alcohol frequency",
                [
                    "Occasionally",
                    "Weekly",
                    "Daily"
                ]
            )

    # =====================================================
    # BASIC VITALS
    # =====================================================

    with st.expander(
        "3. Basic Vitals & History",
        expanded=True
    ):

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120
        )

        high_blood_pressure = st.selectbox(
            "Diagnosed with High Blood Pressure?",
            [
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
            [
                "No",
                "Yes"
            ]
        )

        diabetes = st.selectbox(
            "Diagnosed with Diabetes?",
            [
                "Unknown",
                "No",
                "Yes"
            ]
        )

    # =====================================================
    # ADVANCED TESTS
    # =====================================================

    with st.expander(
        "4. Advanced Clinical Tests",
        expanded=True
    ):

        st.info(
            "Use Unknown or leave numerical values empty "
            "when you do not have the result."
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [
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

        chol_val = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100.0,
            max_value=600.0,
            value=None,
            placeholder="Enter if known"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [
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

        exang = st.selectbox(
            "Exercise Induced Angina",
            [
                "Unknown",
                "No",
                "Yes"
            ]
        )

        oldpeak_val = st.number_input(
            "ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=None,
            step=0.1,
            placeholder="Enter if known"
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [
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

        ca = st.selectbox(
            "Major Vessels Colored by Fluoroscopy",
            [
                "Unknown",
                0,
                1,
                2,
                3
            ]
        )

        thal = st.selectbox(
            "Thalassemia",
            [
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
            "**Blood & Heart Failure Laboratory Results**"
        )

        anaemia = st.selectbox(
            "Anaemia",
            [
                "Unknown",
                "No",
                "Yes"
            ]
        )

        platelets_val = st.number_input(
            "Platelets (kiloplatelets/mL)",
            min_value=10000.0,
            max_value=900000.0,
            value=None,
            step=1000.0,
            placeholder="Enter if known"
        )

        cpk_val = st.number_input(
            "Creatinine Phosphokinase (CPK)",
            min_value=10.0,
            max_value=10000.0,
            value=None,
            placeholder="Enter if known"
        )

        ef_val = st.number_input(
            "Ejection Fraction (%)",
            min_value=10.0,
            max_value=90.0,
            value=None,
            placeholder="Enter if known"
        )

        sc_val = st.number_input(
            "Serum Creatinine (mg/dL)",
            min_value=0.1,
            max_value=10.0,
            value=None,
            step=0.1,
            placeholder="Enter if known"
        )

        ss_val = st.number_input(
            "Serum Sodium (mEq/L)",
            min_value=100.0,
            max_value=160.0,
            value=None,
            step=1.0,
            placeholder="Enter if known"
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
        # HELPERS
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
        # PREPROCESS
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
        # RISK CLASSIFICATION
        # =================================================

        if risk_score < 0.30:

            risk_level = "Low"
            risk_icon = "🟢"

        elif risk_score < 0.70:

            risk_level = "Moderate"
            risk_icon = "🟡"

        else:

            risk_level = "High"
            risk_icon = "🔴"


        # =================================================
        # RESULTS HEADER
        # =================================================

        st.header("📊 Analysis Results")


        # =================================================
        # METRICS
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
                "🟢 **Low Risk:** Your predicted risk "
                "is relatively low."
            )

        elif risk_score < 0.70:

            st.warning(
                "🟡 **Moderate Risk:** Consider reviewing "
                "your health and lifestyle factors."
            )

        else:

            st.error(
                "🔴 **High Risk:** Please consult a "
                "qualified healthcare professional."
            )


        # =================================================
        # LIVE WHAT-IF SIMULATOR
        # =================================================

        st.divider()

        st.subheader(
            "⚡ Live What-If Risk Simulator"
        )

        st.markdown(
            """
            Change the values below and the model will
            automatically recalculate the simulated risk.

            **No Analyze button is required.**

            *This is a model simulation only and does not
            represent actual medical outcomes.*
            """
        )


        # =================================================
        # LIVE CONTROLS
        # =================================================

        live_col1, live_col2 = st.columns(2)

        with live_col1:

            live_age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=int(age),
                step=1,
                key="live_age"
            )

            live_bp = st.slider(
                "Resting Blood Pressure",
                min_value=70,
                max_value=220,
                value=int(trestbps),
                step=1,
                key="live_bp"
            )

            live_chol = st.slider(
                "Cholesterol",
                min_value=100,
                max_value=500,
                value=int(
                    chol_val
                    if chol_val is not None
                    else 200
                ),
                step=1,
                key="live_chol"
            )


        with live_col2:

            live_hr = st.slider(
                "Maximum Heart Rate",
                min_value=60,
                max_value=220,
                value=int(thalach),
                step=1,
                key="live_hr"
            )

            live_smoking = st.selectbox(
                "Smoking Status",
                [
                    "Keep Current",
                    "No",
                    "Yes"
                ],
                key="live_smoking"
            )

            live_diabetes = st.selectbox(
                "Diabetes",
                [
                    "Keep Current",
                    "No",
                    "Yes"
                ],
                key="live_diabetes"
            )


        # =================================================
        # BUILD LIVE DATA
        # =================================================

        live_data = inputs_dict.copy()

        live_data["age"] = live_age

        live_data["trestbps"] = live_bp

        live_data["chol"] = live_chol

        live_data["thalach"] = live_hr


        if live_smoking == "Yes":

            live_data["smoking"] = 1

        elif live_smoking == "No":

            live_data["smoking"] = 0


        if live_diabetes == "Yes":

            live_data["diabetes"] = 1

        elif live_diabetes == "No":

            live_data["diabetes"] = 0


        # =================================================
        # LIVE DATAFRAME
        # =================================================

        live_df = pd.DataFrame(
            [live_data]
        )[feature_names]


        # =================================================
        # LIVE PREPROCESSING
        # =================================================

        live_processed = preprocessor.transform(
            live_df
        )


        # =================================================
        # LIVE PREDICTION
        # =================================================

        live_risk = float(
            model.predict_proba(
                live_processed
            )[0][1]
        )

        live_percentage = live_risk * 100


        # =================================================
        # LIVE CATEGORY
        # =================================================

        if live_risk < 0.30:

            live_level = "Low"
            live_icon = "🟢"

        elif live_risk < 0.70:

            live_level = "Moderate"
            live_icon = "🟡"

        else:

            live_level = "High"
            live_icon = "🔴"


        # =================================================
        # RISK CHANGE
        # =================================================

        live_difference = (
            live_risk - risk_score
        ) * 100


        # =================================================
        # LIVE METRICS
        # =================================================

        st.markdown(
            "### 🔴 Live AI Analysis"
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:

            st.metric(
                "Live Risk",
                f"{live_percentage:.1f}%"
            )

        with r2:

            st.metric(
                "Risk Level",
                f"{live_icon} {live_level}"
            )

        with r3:

            st.metric(
                "Original Risk",
                f"{risk_score * 100:.1f}%"
            )

        with r4:

            st.metric(
                "Change",
                f"{live_difference:+.1f}%"
            )


        # =================================================
        # LIVE GAUGE
        # =================================================

        live_gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=live_percentage,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 52,
                        "color": "#F8FAFC"
                    }
                },

                title={
                    "text": "Live Predicted Risk",
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

        live_gauge.update_layout(

            height=330,

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),

            paper_bgcolor="#0B0F14",

            plot_bgcolor="#0B0F14",

            font={
                "color": "#F8FAFC"
            }
        )

        st.plotly_chart(
            live_gauge,
            use_container_width=True
        )


        # =================================================
        # LIVE MESSAGE
        # =================================================

        if live_difference < -5:

            st.success(
                f"🟢 The simulated profile lowers the "
                f"model's predicted risk by "
                f"{abs(live_difference):.1f} percentage points."
            )

        elif live_difference > 5:

            st.error(
                f"🔴 The simulated profile increases the "
                f"model's predicted risk by "
                f"{live_difference:.1f} percentage points."
            )

        else:

            st.info(
                "🟡 The simulated changes have a relatively "
                "small effect on the model's prediction."
            )


        # =================================================
        # LIVE COMPARISON CHART
        # =================================================

        comparison_fig = go.Figure()

        comparison_fig.add_trace(

            go.Bar(

                x=[
                    "Original Risk",
                    "Live Risk"
                ],

                y=[
                    risk_score * 100,
                    live_risk * 100
                ],

                text=[
                    f"{risk_score * 100:.1f}%",
                    f"{live_risk * 100:.1f}%"
                ],

                textposition="auto"
            )
        )

        comparison_fig.update_layout(

            title={
                "text": "Original vs Live Prediction",
                "font": {
                    "color": "#F8FAFC"
                }
            },

            yaxis={
                "title": "Risk (%)",
                "range": [0, 100],
                "gridcolor": "#263241"
            },

            height=350,

            paper_bgcolor="#0B0F14",

            plot_bgcolor="#151B23",

            font={
                "color": "#F8FAFC"
            },

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=30
            )
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )


        # =================================================
        # LIVE PROFILE
        # =================================================

        st.markdown(
            "### 📊 Current Simulation Profile"
        )

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.metric(
                "Age",
                live_age
            )

        with s2:

            st.metric(
                "Blood Pressure",
                f"{live_bp} mmHg"
            )

        with s3:

            st.metric(
                "Cholesterol",
                f"{live_chol} mg/dL"
            )

        with s4:

            st.metric(
                "Max Heart Rate",
                f"{live_hr} bpm"
            )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        st.divider()

        st.subheader(
            "🧠 What Drove This Prediction?"
        )

        st.markdown(
            """
            SHAP explains which features contributed most
            strongly to the model's prediction.
            """
        )


        if explainer is not None:

            try:

                with st.spinner(
                    "Analyzing prediction factors..."
                ):

                    shap_result = explainer.shap_values(
                        processed_input
                    )

                    if isinstance(
                        shap_result,
                        list
                    ):

                        shap_vals = np.asarray(
                            shap_result[-1]
                        )[0]

                    else:

                        shap_vals = np.asarray(
                            shap_result
                        )

                        if shap_vals.ndim == 2:

                            shap_vals = shap_vals[0]

                        elif shap_vals.ndim == 3:

                            shap_vals = shap_vals[0, :, -1]


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
                            "Maximum Heart Rate",

                        "exang":
                            "Exercise Angina",

                        "oldpeak":
                            "ST Depression",

                        "slope":
                            "ST Segment Slope",

                        "ca":
                            "Major Vessels",

                        "thal":
                            "Thalassemia",

                        "anaemia":
                            "Anaemia",

                        "creatinine_phosphokinase":
                            "CPK Level",

                        "diabetes":
                            "Diabetes",

                        "ejection_fraction":
                            "Ejection Fraction",

                        "high_blood_pressure":
                            "High Blood Pressure",

                        "platelets":
                            "Platelets",

                        "serum_creatinine":
                            "Serum Creatinine",

                        "serum_sodium":
                            "Serum Sodium",

                        "smoking":
                            "Smoking",

                    }


                    feature_impacts = {}

                    for i, feature in enumerate(
                        feature_names
                    ):

                        if i < len(shap_vals):

                            feature_impacts[
                                readable_names.get(
                                    feature,
                                    feature
                                )
                            ] = float(
                                shap_vals[i]
                            )


                    sorted_impacts = sorted(
                        feature_impacts.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:6]


                    top_factors_dict = {
                        key: value
                        for key, value
                        in sorted_impacts
                    }


                    factors = [
                        item[0]
                        for item in reversed(
                            sorted_impacts
                        )
                    ]

                    impacts = [
                        item[1]
                        for item in reversed(
                            sorted_impacts
                        )
                    ]


                    colors = [

                        "#ff4b4b"
                        if value > 0
                        else "#00cc96"

                        for value in impacts
                    ]


                    shap_fig = go.Figure(

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


                    shap_fig.update_layout(

                        title={
                            "text":
                                "Top 6 Contributing Factors",
                            "font": {
                                "color":
                                    "#F8FAFC"
                            }
                        },

                        xaxis_title=
                            "Impact on Prediction",

                        height=400,

                        paper_bgcolor="#0B0F14",

                        plot_bgcolor="#0B0F14",

                        font={
                            "color":
                                "#F8FAFC"
                        },

                        xaxis={
                            "zeroline": True,
                            "zerolinewidth": 2,
                            "zerolinecolor":
                                "#64748B",
                            "gridcolor":
                                "#263241"
                        },

                        yaxis={
                            "gridcolor":
                                "#263241"
                        }
                    )


                    st.plotly_chart(
                        shap_fig,
                        use_container_width=True
                    )

            except Exception as shap_error:

                st.info(
                    f"SHAP explanation unavailable: "
                    f"{shap_error}"
                )

                top_factors_dict = {}


        else:

            st.info(
                "SHAP explainer is not available."
            )

            top_factors_dict = {}


        # =================================================
        # GROQ AI HEALTH ASSISTANT
        # =================================================

        st.divider()

        st.subheader(
            "🤖 AI Health Assistant"
        )

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


        with st.spinner(
            "Generating personalized AI explanation..."
        ):

            try:

                explanation = generate_explanation(

                    risk_score,

                    top_factors_dict,

                    original_inputs
                )

                st.markdown(
                    explanation
                )

            except Exception as llm_error:

                explanation = (
                    "AI explanation could not be generated "
                    "at this time."
                )

                st.warning(
                    f"AI Assistant unavailable: {llm_error}"
                )


        # =================================================
        # SAVE TO SUPABASE
        # =================================================

        with st.spinner(
            "Saving prediction to history..."
        ):

            try:

                inserted = insert_prediction(

                    st.session_state.user_id,

                    original_inputs,

                    risk_score,

                    top_factors_dict,

                    explanation
                )

                if not inserted:

                    st.caption(
                        "Database not connected or "
                        "prediction could not be saved."
                    )

            except Exception as db_error:

                st.caption(
                    f"History save unavailable: {db_error}"
                )


# =========================================================
# PATIENT HISTORY
# =========================================================

st.divider()

with st.expander(
    "📋 View Patient History"
):

    try:

        history = get_user_history(
            st.session_state.user_id
        )

        if history:

            for record in history:

                created_at = str(
                    record.get(
                        "created_at",
                        ""
                    )
                )

                risk_value = float(
                    record.get(
                        "risk_score",
                        0
                    )
                )

                st.markdown(
                    f"""
                    **Date:** {created_at[:10]}

                    **Risk Score:** {risk_value:.1%}

                    ---
                    """
                )

        else:

            st.write(
                "No previous history found."
            )

    except Exception as history_error:

        st.info(
            f"Patient history unavailable: "
            f"{history_error}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🫀 HealthGuard AI • Machine Learning + Explainable AI "
    "+ Personalized Health Assistant"
)

st.caption(
    "For educational and research purposes only. "
    "Not a medical diagnosis system."
)