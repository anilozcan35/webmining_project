import streamlit as st

def get_prediction_records(key_start=0):
    """Get the prediction records from the user input."""
    age = st.text_input(label="Age", value=26.0, key=key_start)
    # gender = st.text_input(label="Gender", value="M", key=key_start + 1)
    gender = st.selectbox("Gender", ("F", "M"), key=key_start + 1)
    height_cm = st.text_input(label="HeightCm", value=170, key=key_start + 2)
    weight_kg = st.text_input(label="WeightKg", value=55.8, key=key_start + 3)
    fat = st.text_input(label="Fat", value=15.7, key=key_start + 4)
    diastolic = st.text_input(label="Diastolic", value=77.0, key=key_start + 5)
    systolic = st.text_input(label="Systolic", value=126.0, key=key_start + 6)
    gripForce = st.text_input(label="GripForce", value=36.4, key=key_start + 7)
    forward_cm = st.text_input(label="ForwardCm", value=16.3, key=key_start + 8)
    sit_ups = st.text_input(label="SitUps", value=53.0, key=key_start + 9)
    jump_cm = st.text_input(label="JumpCm", value=29.0, key=key_start + 10)
    classs = "A"
    prediction_record = [float(age),
                         gender,
                         float(height_cm),
                         float(weight_kg),
                         float(fat),
                         float(diastolic),
                         float(systolic),
                         float(gripForce),
                         float(forward_cm),
                         float(sit_ups),
                         float(jump_cm),
                         classs]
    return prediction_record
