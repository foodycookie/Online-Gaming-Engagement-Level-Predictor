import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load data
feature_columns = joblib.load('model/feature_columns.pkl')
scaler          = joblib.load('model/scaler.pkl')
dt_model        = joblib.load('model/Decision Tree.pkl')
rf_model        = joblib.load('model/Random Forest.pkl')
xgb_model       = joblib.load('model/XGBoost.pkl')

label_map = {0: 'Low', 1: 'Medium', 2: 'High'}

# UI
st.title("Online Gaming Engagement Level Predictor")

model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest", "XGBoost"])

with st.form("input_form"):
    st.subheader("Player Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=49, value=25, step=1)

        gender = st.radio("Gender", options=["Male", "Female"])

        location = st.selectbox("Location", ["Asia", "Europe", "USA", "Other"])

        game_genre = st.selectbox("Game Genre", ["Action", "RPG", "Simulation", "Sports", "Strategy"])

        game_difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])

    with col2:
        play_time_hours = st.number_input("Play Time Hours", min_value=0.0, max_value=24.0, value=10.0, step=0.5)

        in_game_purchases = st.selectbox("In-Game Purchases", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        sessions_per_week = st.number_input("Sessions Per Week", min_value=0, max_value=19, value=5, step=1)

        avg_session_duration = st.number_input("Avg Session Duration (Minutes)", min_value=10, max_value=179, value=100, step=5)

        player_level = st.number_input("Player Level", min_value=1, max_value=99, value=20, step=1)

        achievements_unlocked = st.number_input("Achievements Unlocked", min_value=0, max_value=49, value=15, step=1)

    submitted = st.form_submit_button("Predict Engagement Level", use_container_width=True)

# Prepare input for model
if submitted:
    location_europe = 1 if location == "Europe" else 0
    location_other  = 1 if location == "Other"  else 0
    location_usa    = 1 if location == "USA"    else 0

    genre_rpg        = 1 if game_genre == "RPG"        else 0
    genre_simulation = 1 if game_genre == "Simulation" else 0
    genre_sports     = 1 if game_genre == "Sports"     else 0
    genre_strategy   = 1 if game_genre == "Strategy"   else 0
    
    difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    
    gender_encoded = 1 if gender == "Male" else 0
    
    input_df = pd.DataFrame({
        "Age"                        : [age],
        "Gender"                     : [gender_encoded],
        "PlayTimeHours"              : [play_time_hours],
        "InGamePurchases"            : [in_game_purchases],
        "GameDifficulty"             : [difficulty_map[game_difficulty]],
        "SessionsPerWeek"            : [sessions_per_week],
        "AvgSessionDurationMinutes"  : [avg_session_duration],
        "PlayerLevel"                : [player_level],
        "AchievementsUnlocked"       : [achievements_unlocked],
        "Location_Europe"            : [location_europe],
        "Location_Other"             : [location_other],
        "Location_USA"               : [location_usa],
        "GameGenre_RPG"              : [genre_rpg],
        "GameGenre_Simulation"       : [genre_simulation],
        "GameGenre_Sports"           : [genre_sports],
        "GameGenre_Strategy"         : [genre_strategy],
    })
    
    # Align columns to training order
    input_df = input_df[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Select model
    if model_choice == "Decision Tree":
        model = dt_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    prediction      = model.predict(input_scaled)[0]
    probabilities   = model.predict_proba(input_scaled)[0]
    predicted_label = label_map[prediction]

    # Result
    st.divider()
    st.subheader("Prediction Result")

    if predicted_label == "High":
        st.success(f"Engagement Level: **{predicted_label}**")
    elif predicted_label == "Medium":
        st.warning(f"Engagement Level: **{predicted_label}**")
    else:
        st.error(f"Engagement Level: **{predicted_label}**")