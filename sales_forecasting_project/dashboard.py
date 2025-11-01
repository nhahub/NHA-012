import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Walmart Sales Forecaster",
    layout="centered",
    initial_sidebar_state="auto",
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("Walmart Sales Forecasting")
st.markdown("Enter the details below to get a sales forecast.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Store & Dept")
    store = st.number_input("Store ID", min_value=1, max_value=45, value=1, step=1)
    dept = st.number_input("Department ID", min_value=1, max_value=99, value=1, step=1)
    
    st.subheader("Store Details")
    store_type = st.selectbox("Store Type", options=["A", "B", "C"], index=0)
    size = st.number_input("Store Size", min_value=30000, max_value=250000, value=151315, step=1000)

with col2:
    st.subheader("Date Information")
    year = st.number_input("Year", min_value=2010, max_value=2013, value=2011, step=1)
    week = st.number_input("Week of Year", min_value=1, max_value=52, value=42, step=1)
    day = st.number_input("Day of Month", min_value=1, max_value=31, value=28, step=1)
    
    is_holiday = st.checkbox("Is it a Holiday?")

st.divider()

if st.button("📈 Predict Sales", use_container_width=True, type="primary"):

    payload = {
        "Store": store,
        "Dept": dept,
        "IsHoliday": is_holiday,
        "Size": size,
        "Type": store_type,
        "Year": year,
        "Week": week,
        "Day": day
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        prediction_data = response.json()
        forecasted_sales = prediction_data.get("forecasted_sales")

        if forecasted_sales is not None:
            st.success(f"**Forecasted Sales: ${forecasted_sales:,.2f}**")
            
            with st.expander("Show Prediction Details"):
                st.json(prediction_data)
        else:
            st.error(f"Error in API response: {prediction_data.get('error')}")

    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to the API at {API_URL}. Is the API server running?")
    except Exception as e:
        st.error(f"An error occurred: {e}")