"""
Spaceship Titanic – Streamlit App
Serve the trained Logistic Regression classifier via a web UI.
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from pre_processing import feature_engineering, CATEGORICAL_COLS, NUMERICAL_COLS


# Load preprocessor and model
preprocessor = joblib.load(Path(__file__).parent / "artifacts/preprocessor.pkl")
model        = joblib.load(Path(__file__).parent / "artifacts/model.pkl")


def make_prediction(raw_input: dict) -> str:
    df = pd.DataFrame([raw_input])
    df = feature_engineering(df)

    label_encoders  = preprocessor["label_encoders"]
    num_medians     = preprocessor["num_medians"]
    feature_columns = preprocessor["feature_columns"]

    # Fill missing
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(num_medians.get(col, 0))

    # Encode categoricals (handle unseen labels)
    for col in CATEGORICAL_COLS:
        le = label_encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
        df[col] = le.transform(df[col])

    X = df[feature_columns]
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return bool(pred), prob


def main():
    st.title("ASG 04 MD - Valentino - Spaceship Titanic Model Deployment")

    st.subheader("Passenger Details")
    col1, col2 = st.columns(2)

    with col1:
        passenger_id = st.text_input("Passenger ID", value="6767_67", help="Format: gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group.")
        home_planet  = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
        cryo_sleep   = st.radio("Cryo Sleep", ["False", "True"], help="Was the passenger elected to be put into suspended animation for the duration of the voyage? Passengers in cryosleep are confined to their cabins.")
        cabin        = st.text_input("Cabin", value="B/67/P", help="Format: deck/num/side, where side can be either P for Port or S for Starboard.")
        destination  = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
        age          = st.number_input("Age", min_value=0, max_value=100, value=67)
        vip          = st.radio("VIP", ["False", "True"], help="Whether the passenger has paid for special VIP service during the voyage.")

    with col2:
        name          = st.text_input("Name", value="Pria Solo", help="The first and last names of the passenger.")
        room_service  = st.number_input("Room Service",   min_value=0, value=0, step=10, help="RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.")
        food_court    = st.number_input("Food Court",     min_value=0, value=0, step=10)
        shopping_mall = st.number_input("Shopping Mall",  min_value=0, value=0, step=10)
        spa           = st.number_input("Spa",            min_value=0, value=0, step=10)
        vr_deck       = st.number_input("VR Deck",        min_value=0, value=0, step=10)

    if st.button("Make Prediction"):
        raw_input = {
            "PassengerId":  passenger_id,
            "HomePlanet":   home_planet,
            "CryoSleep":    cryo_sleep == "True",
            "Cabin":        cabin,
            "Destination":  destination,
            "Age":          age,
            "VIP":          vip == "True",
            "RoomService":  room_service,
            "FoodCourt":    food_court,
            "ShoppingMall": shopping_mall,
            "Spa":          spa,
            "VRDeck":       vr_deck,
            "Name":         name,
        }
        transported, prob = make_prediction(raw_input)

        if transported:
            st.success(f"Predicted: TRANSPORTED (probability: {prob:.1%})")
        else:
            st.warning(f"Predicted: NOT TRANSPORTED (probability: {prob:.1%})")


if __name__ == "__main__":
    main()
