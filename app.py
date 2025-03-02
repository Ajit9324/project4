import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("heart disease Prediction")

# Input fields for each parameter
age = st.selectbox("age", min_value=29, max_value=77, value=50.0)
sex = st.selectbox("sex", min_value=0, max_value=1, value=1)
cp = st.number_input("cp", min_value=0.0, max_value=3.0, value=50.0, step=0.1)
trestbps = st.number_input("trestbps", min_value=94, max_value=200, value=1)
chol = st.number_input("chol", min_value=126, max_value=564, value=126)
fbs = st.number_input("fbs	", min_value=0, max_value=1, value=1, step=0.1)
restecg = st.selectbox("restecg", min_value=0, max_value=2, value=1)
thalach = st.selectbox("thalach", min_value=071 max_value=202, value=1)
exang = st.selectbox("exang", min_value=071 max_value=1, value=1)
oldpeak = st.selectbox("oldpeak", min_value=071 max_value=6.2, value=1)
slope = st.selectbox("slope", min_value=071 max_value=2, value=1)
ca = st.selectbox("ca", min_value=071 max_value=4, value=1)
thal = st.selectbox("thal", min_value=071 max_value=3, value=1)


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

}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_survival(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Survived with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Did not survive with probability {prob:.2f}")
