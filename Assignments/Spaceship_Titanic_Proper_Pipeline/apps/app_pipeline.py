import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.config import ARTIFACT_PIPELINE, SPENDING_COLS
from src.utils.io import load_artifact
from src.data.loader import feature_engineering


@st.cache_resource
def load_pipeline():
    try:
        return load_artifact(ARTIFACT_PIPELINE)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


def make_prediction(raw_input: dict):
    model = load_pipeline()
    df    = pd.DataFrame([raw_input])
    df    = feature_engineering(df)
    pred  = model.predict(df)[0]
    prob  = model.predict_proba(df)[0][1]
    return bool(pred), float(prob)


def main():
    st.title("ASG 05 MD - Valentino - Spaceship Titanic Pipeline Model Deployment")

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
