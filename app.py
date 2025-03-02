import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_heartattack(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("heart disease Prediction")

# Input fields for each parameter
age = st.number_input("age", min_value=29, max_value=77, value=50)
sex = st.number_input("sex", min_value=0, max_value=1, value=1)
cp = st.number_input("cp", min_value=0, max_value=3, value=0, step=1)
trestbps = st.number_input("trestbps", min_value=94, max_value=200, value=94)
chol = st.number_input("chol", min_value=126, max_value=564, value=126)
fbs = st.number_input("fbs	", min_value=0, max_value=1, value=1, step=1)
restecg = st.number_input("restecg", min_value=0, max_value=2, value=1)
thalach = st.number_input("thalach", min_value=71, max_value=202, value=71)
exang = st.number_input("exang", min_value=0, max_value=1, value=1)
oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=6.2, value=1.0)
slope = st.number_input("slope", min_value=0, max_value=2, value=1)
ca = st.number_input("ca", min_value=0, max_value=4, value=1)
thal = st.number_input("thal", min_value=0, max_value=3, value=1)


# Map the gender and embarked values to numeric
#gender_map = {'male': 0, 'female': 1}
#mbarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Create the input dictionary for prediction
input_data = {
    'age': age ,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'thal': thal,
    'ca' : ca,

}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_heartattack(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: heartattack is possible with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: heartattack is possible with probability {prob:.2f}")
