import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
# Assuming we have a pre-trained model
import joblib  # For loading the saved model

# Load the pre-trained model (you'd need to have this model file)
try:
    model = joblib.load('delivery_time_model.pkl')
except:
    st.error("Model file not found. Please ensure 'delivery_time_model.pkl' exists.")

# Title and description
st.title("Timelytics: Order to Delivery Time Prediction")
st.write("Enter order details to predict the expected delivery time")

# Create input form
with st.form(key='prediction_form'):
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        product_category = st.selectbox(
            "Product Category",
            ["Electronics", "Clothing", "Books", "Home Goods", "Sports"]
        )
        
        shipping_method = st.selectbox(
            "Shipping Method",
            ["Standard", "Express", "Overnight"]
        )
        
    with col2:
        customer_location = st.selectbox(
            "Customer Location",
            ["North America", "Europe", "Asia", "Australia", "South America"]
        )
        
        order_date = st.date_input("Order Date", datetime.now())
    
    # Additional numerical inputs
    order_quantity = st.number_input("Order Quantity", min_value=1, value=1)
    
    submit_button = st.form_submit_button(label='Predict Delivery Time')

# Prediction logic
if submit_button:
    # Prepare input data for the model
    input_data = {
        'product_category': product_category,
        'shipping_method': shipping_method,
        'customer_location': customer_location,
        'order_quantity': order_quantity,
        # Convert date to numerical features if needed
        'order_day': order_date.day,
        'order_month': order_date.month,
        'order_year': order_date.year
    }
    
    # Convert to DataFrame (assuming model expects this format)
    input_df = pd.DataFrame([input_data])
    
    # Add any necessary preprocessing steps here
    # For example: one-hot encoding categorical variables
    categorical_cols = ['product_category', 'shipping_method', 'customer_location']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Ensure all columns from training are present
    # This is a placeholder - you'd need to adjust based on your actual model
    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        
        # Display results
        st.success("Prediction Successful!")
        st.write(f"Estimated Delivery Time: {prediction:.1f} days")
        
        # Calculate estimated delivery date
        estimated_date = order_date + pd.Timedelta(days=prediction)
        st.write(f"Estimated Delivery Date: {estimated_date.strftime('%Y-%m-%d')}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add some additional information
st.markdown("""
### How to Use
1. Select the product category from the dropdown
2. Choose the shipping method
3. Select the customer location
4. Enter the order quantity
5. Pick the order date
6. Click 'Predict Delivery Time' to get the estimation

*Note: This prediction is based on historical data and may vary due to external factors.*
""")
