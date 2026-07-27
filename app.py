import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Enterprise Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("customer_churn_model.pkl")
features = joblib.load("model_features.pkl")

# -------------------------
# Title
# -------------------------
st.title("📊 Enterprise Customer Intelligence Platform")

st.markdown("""
### AI Powered Customer Churn Prediction System

Predict customer churn using a Machine Learning model.

Fill in the customer information from the sidebar and click **Predict Churn**.
""")

st.divider()

# -------------------------
# Model Information
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Random Forest")

with col2:
    st.metric("Features", len(features))

with col3:
    st.metric("Prediction", "Binary Classification")

st.divider()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Customer Information")

st.sidebar.info("""
Enter customer details below.

Click **Predict Churn** to analyze the customer.
""")

user_input = {}

for feature in features:
    value = st.sidebar.number_input(
        feature,
        value=0.0,
        format="%.2f"
    )
    user_input[feature] = value

# -------------------------
# Prediction
# -------------------------
if st.sidebar.button("Predict Churn"):

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        if prediction == 1:
            st.error("⚠ High Churn Risk")
        else:
            st.success("✅ Low Churn Risk")

    with c2:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with c3:
        risk = (
            "High" if probability >= 0.70
            else "Medium" if probability >= 0.40
            else "Low"
        )

        st.metric(
            "Risk Level",
            risk
        )

    st.write("### Probability Score")

    st.progress(float(probability))

    st.write(f"Probability: **{probability:.2%}**")

    st.divider()

    st.subheader("Business Recommendation")

    if probability >= 0.70:
        st.error("""
🔴 High Risk Customer

Recommended Actions:

• Offer a special discount

• Send a personalized email

• Contact customer support

• Give loyalty rewards
""")

    elif probability >= 0.40:
        st.warning("""
🟡 Medium Risk Customer

Recommended Actions:

• Send promotional offers

• Monitor purchase behavior

• Recommend popular products
""")

    else:
        st.success("""
🟢 Low Risk Customer

Recommended Actions:

• Keep engaged with newsletters

• Upsell premium products

• Maintain loyalty benefits
""")

# -------------------------
# Footer
# -------------------------
st.divider()

st.caption(
    "Developed by Karan Singh | Enterprise Customer Intelligence Platform | Machine Learning + Streamlit"
)