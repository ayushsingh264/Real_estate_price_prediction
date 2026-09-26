import os
import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Real Estate Price Prediction System")
st.caption("Industry-ready capstone dashboard")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    if health["model_loaded"]:
        st.success("API and model are online.")
    else:
        st.warning("API is online, but the model is not loaded.")
except requests.RequestException:
    st.error(f"Cannot connect to API: {API_URL}")

st.divider()

left, right = st.columns(2)

with left:
    area = st.number_input("Area (sq ft)", min_value=100.0, max_value=50000.0, value=2500.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)

with right:
    age = st.number_input("Property Age (years)", min_value=0, max_value=300, value=10)
    location = st.selectbox("Location", ["Rural", "Suburb", "City Center"])
    property_type = st.selectbox("Property Type", ["House", "Villa", "Apartment"])

if st.button("Predict Property Price", type="primary"):
    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "location": location,
        "property_type": property_type,
    }

    try:
        response = requests.post(
            f"{API_URL}/api/v1/predict",
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()

        prediction = result["predicted_price"]
        lower = result["confidence_interval"]["lower_bound"]
        upper = result["confidence_interval"]["upper_bound"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price", f"₹{prediction:,.0f}")
        c2.metric("Lower Bound", f"₹{lower:,.0f}")
        c3.metric("Upper Bound", f"₹{upper:,.0f}")

        fig = go.Figure(
            go.Bar(
                x=["Lower", "Estimated", "Upper"],
                y=[lower, prediction, upper],
            )
        )
        fig.update_layout(
            title="Estimated Price Range",
            xaxis_title="Estimate",
            yaxis_title="Price (INR)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Model: {result['selected_model']} | "
            f"Version: {result['model_version']}"
        )

    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
