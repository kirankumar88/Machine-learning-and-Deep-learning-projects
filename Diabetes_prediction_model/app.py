import streamlit as st
import numpy as np
import pickle

# Load trained logistic regression model
model = pickle.load(open('Diabetes_prediction_model.pkl', 'rb'))

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🩺 Diabetes Risk Predictor",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------- HEADER --------------------
st.markdown(
    """
    <style>
    .title {
        font-size:36px !important;
        color:#2E86C1;
        text-align:center;
        font-weight:700;
    }
    .sub {
        text-align:center;
        color:#1B2631;
        font-size:18px;
    }
    .footer {
        text-align:center;
        color:gray;
        font-size:13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">🩺 Diabetes Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Powered by Logistic Regression | Developed by Kiran</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.header("👤 Patient Information")
st.sidebar.markdown("Provide the following medical details:")

Pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, 1)
Glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
BloodPressure = st.sidebar.slider("Blood Pressure (mmHg)", 0, 150, 70)
SkinThickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
Insulin = st.sidebar.slider("Insulin Level", 0, 900, 80)
BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
DPF = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.sidebar.slider("Age", 10, 100, 35)

# -------------------- PREDICTION LOGIC --------------------
if st.sidebar.button("🔍 Predict Diabetes Risk"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### 📊 Prediction Result:")
    st.markdown("---")

    if prediction[0] == 1:
        st.error(f"**🧬 The patient is likely to have Diabetes.**")
        st.metric(label="Probability of Diabetes", value=f"{probability*100:.2f}%")
        st.progress(min(1.0, probability))
    else:
        st.success(f"**✅ The patient is not likely to have Diabetes.**")
        st.metric(label="Probability of Diabetes", value=f"{probability*100:.2f}%")
        st.progress(min(1.0, probability))

    st.markdown("---")
    st.markdown("### 🧠 Model Insights")
    st.info("""
    - Logistic Regression model trained on the PIMA Diabetes dataset  
    - Data preprocessed with outlier removal and median imputation  
    - AUC Score: ~0.76 (balanced performance)  
    """)

# -------------------- ABOUT SECTION --------------------
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This app predicts the likelihood of diabetes based on clinical indicators.
    It uses a **Logistic Regression** model trained on the **PIMA Indian Diabetes dataset**.

    **Input Features:**
    - Pregnancies  
    - Glucose Level  
    - Blood Pressure  
    - Skin Thickness  
    - Insulin Level  
    - BMI (Body Mass Index)  
    - Diabetes Pedigree Function  
    - Age

    **Output:**
    - Predicted class (Diabetic / Non-Diabetic)
    - Probability score indicating risk level
    """)

st.markdown("---")
st.markdown('<p class="footer">© 2025 Developed by Kiran | Powered by Streamlit</p>', unsafe_allow_html=True)
