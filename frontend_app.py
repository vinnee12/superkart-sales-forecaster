import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")

st.title("🛒 SuperKart Sales Forecasting System")
st.write("Analyze store sales parameters and forecast your upcoming quarter revenues with precision.")

# Sidebar config for backend URL
backend_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://backend:7860",
    help="Enter the host and port where your Flask backend container is listening."
)

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Real-time Single Product Forecast")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.5, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        mrp = st.number_input("Product MRP ($)", min_value=1.0, max_value=500.0, value=150.0, step=0.5)
        
    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_city = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Food Mart", "Departmental Store", "Supermarket Type1", "Supermarket Type2"])
        
    with col3:
        product_id_char = st.selectbox("Product ID Characters", ["FD", "DR", "NC"], help="FD: Food, DR: Drink, NC: Non-Consumable")
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=100, value=15, step=1)
        product_category = st.selectbox("Product Category", ["Perishables", "Non Perishables"])

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": product_category
        }
        
        try:
            with st.spinner("Calculating forecast..."):
                response = requests.post(f"{backend_url}/v1/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    predicted_sales = result.get("Sales", 0.0)
                    st.success(f"🎯 **Predicted Total Sales Revenue: ${predicted_sales:,.2f}**")
                else:
                    st.error(f"Backend returned an error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Could not connect to the backend server: {e}")

with tab2:
    st.subheader("Bulk Batch Predictions")
    st.write("Upload a CSV file with your product properties to generate a batch of forecasts.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded dataset", df.head())
        
        if st.button("Run Batch Inference"):
            try:
                with st.spinner("Executing bulk inference pipeline..."):
                    files = {'file': uploaded_file.getvalue()}
                    response = requests.post(f"{backend_url}/v1/predictbatch", files=files, timeout=30)
                    if response.status_code == 200:
                        predictions = response.json()
                        preds_series = pd.Series(predictions).astype(float)
                        df['Predicted_Sales'] = preds_series.values
                        
                        st.success("🎉 Batch Forecasting Completed Successfully!")
                        st.dataframe(df)
                        
                        # Download result as CSV
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_data,
                            file_name="SuperKart_Sales_Predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Backend returned error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Could not connect to the backend server: {e}")
