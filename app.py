import streamlit as st
import tensorflow as tf
import numpy as np

# Set page configuration
st.set_page_config(page_title="Churn Prediction", layout="centered")

# Custom CSS for a modern and elegant UI
st.markdown("""
    <style>
        body {
            background-color: #ffffff !important;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2E3B55;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #007BFF !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px 24px;
            font-size: 18px;
            font-weight: bold;
        }
        .stRadio label {
            font-size: 16px;
        }
        .stNumberInput, .stSelectbox, .stSlider {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("🔮 Churn Prediction")
st.markdown("<p style='text-align: center; font-size:18px; color:#555;'>This application predicts whether a customer is likely to churn based on their details.</p>", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

if model:
    

    # Customer Details Section
    st.subheader("📋 Enter Customer Details")

    # Input fields with stylish layout
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=850, value=650)
            age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
            tenure = st.number_input("📅 Tenure (years)", min_value=0, max_value=10, value=5)
            balance = st.number_input("💰 Balance", min_value=0.0, max_value=500000.0, value=10000.0)
            num_products = st.slider("🛍️ Number of Products", min_value=1, max_value=4, value=1)

        with col2:
            has_credit_card = st.radio("💳 Has Credit Card?", ["Yes", "No"])
            is_active_member = st.radio("✅ Is Active Member?", ["Yes", "No"])
            estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
            gender = st.radio("⚤ Gender", ["Male", "Female"])
            geography = st.selectbox("🌍 Geography (Country)", ["France", "Germany", "Spain"])

    # Convert categorical inputs to numerical values
    has_credit_card = 1 if has_credit_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    gender = 1 if gender == "Male" else 0  # Male = 1, Female = 0

    # One-Hot Encoding for Geography
    geo_france = 1 if geography == "France" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Prepare input data (Now contains 12 features)
    input_data = np.array([[credit_score, age, tenure, balance, num_products, has_credit_card, is_active_member,
                             estimated_salary, gender, geo_france, geo_germany, geo_spain]])

    # Predict Button with Modern Styling
    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_data)
        churn_probability = prediction[0][0]
        st.subheader(f"🧐 Churn Probability: **{churn_probability:.2f}**")

        if churn_probability > 0.5:
            st.warning("🚨 **The customer is likely to churn!**")
        else:
            st.success("✅ **The customer is unlikely to churn.**")

else:
    st.error("⚠️ Model could not be loaded. Please check the file.")
