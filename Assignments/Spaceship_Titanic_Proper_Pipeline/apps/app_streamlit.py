"""
Run AFTER starting the FastAPI server:
  uvicorn apps.spaceship_fastapi:app --reload
Then run:
  streamlit run apps/app_streamlit.py
"""

import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000/predict"


def make_prediction(features: dict):
    try:
        response = requests.post(FASTAPI_URL, json=features, timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI server. Start it first:\n\n"
            "`uvicorn apps.spaceship_fastapi:app --reload`"
        )
        return None


def main():
    st.title("ASG 07 MD - Valentino - Spaceship Titanic FastAPI Model Deployment")

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
        features = {
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

        result = make_prediction(features)
        if result is not None:
            if result["transported"]:
                st.success(f"Prediction: {result['message']}  (probability: {result['probability']:.1%})")
            else:
                st.warning(f"Prediction: {result['message']}  (probability: {result['probability']:.1%})")


if __name__ == "__main__":
    main()
