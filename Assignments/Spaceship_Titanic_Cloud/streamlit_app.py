"""
Streamlit UI for the Spaceship Titanic classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "spaceship-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(passenger: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [passenger]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Spaceship Titanic Classifier")
st.title("ASG 10 MD - Valentino - Spaceship Titanic Cloud Model Deployment")

st.subheader("Passenger Details")
col1, col2 = st.columns(2)

with col1:
    home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
    cryo_sleep  = st.radio("Cryo Sleep", ["False", "True"], help="Was the passenger elected to be put into suspended animation for the duration of the voyage? Passengers in cryosleep are confined to their cabins.")
    destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    age         = st.number_input("Age", min_value=0, max_value=100, value=67)
    vip         = st.radio("VIP", ["False", "True"], help="Whether the passenger has paid for special VIP service during the voyage.")
    deck        = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G", "T"], index=5)
    cabin_num   = st.number_input("Cabin Number", min_value=0, value=3, step=1)
    side        = st.selectbox("Cabin Side", ["P", "S"], index=1)

with col2:
    room_service  = st.number_input("Room Service",   min_value=0, value=0, step=10, help="RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.")
    food_court    = st.number_input("Food Court",     min_value=0, value=0, step=10)
    shopping_mall = st.number_input("Shopping Mall",  min_value=0, value=0, step=10)
    spa           = st.number_input("Spa",            min_value=0, value=0, step=10)
    vr_deck       = st.number_input("VR Deck",        min_value=0, value=0, step=10)

if st.button("Predict", type="primary"):
    passenger = {
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Cabin": f"{deck}/{int(cabin_num)}/{side}",
        "Destination": destination,
        "Age": float(age),
        "VIP": vip,
        "RoomService": float(room_service),
        "FoodCourt": float(food_court),
        "ShoppingMall": float(shopping_mall),
        "Spa": float(spa),
        "VRDeck": float(vr_deck),
    }
    try:
        result = invoke_endpoint(passenger)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]

        st.success(f"Predicted outcome: **{label}**")
        st.write("Class probabilities:")
        prob_df = pd.DataFrame(
            {"probability": probs},
            index=["not_transported", "transported"],
        )
        st.bar_chart(prob_df)
