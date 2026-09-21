
import streamlit as st
import pandas as pd
import joblib

# Load model and original training columns
saved_data = joblib.load("car_price_model.pkl")

model = saved_data["model"]
model_columns = saved_data["columns"]

st.title("🚗 Car Price Prediction")
st.write("Enter the car details to estimate its selling price.")

# -------------------------
# User inputs
# -------------------------

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2020,
    value=2018,
    step=1
)

age = 2020 - year

st.write(f"Car Age: {age} years")

present_price = st.number_input(
    "Present Price (in lakhs)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000,
    step=1000
)

past_owners = st.selectbox(
    "Previous Owners",
    [0, 1, 2, 3]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# -------------------------
# Create input dataframe
# -------------------------

if st.button("Predict Selling Price"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Present_Price(lacs)": present_price,
        "Kms_Driven": kms_driven,
        "Past_Owners": past_owners
    }])

    # Add categorical dummy columns
    for col in model_columns:

        if col.startswith("Fuel_Type_"):
            category = col.replace("Fuel_Type_", "")
            input_df[col] = int(fuel_type == category)

        elif col.startswith("Seller_Type_"):
            category = col.replace("Seller_Type_", "")
            input_df[col] = int(seller_type == category)

        elif col.startswith("Transmission_"):
            category = col.replace("Transmission_", "")
            input_df[col] = int(transmission == category)

    # Ensure exact training column order
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Selling Price: ₹{prediction:.2f} lakhs"
    )