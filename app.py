import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf


st.title("Passenger Survival Chance in the Titanic Journey")


pclass = st.slider(
    'Enter the Passenger class for the user',
    1, 3
)

sex = st.selectbox(
    "Enter the Passenger Gender",
    ['male', 'female']
)

sibsp = st.slider(
    'Enter the Passenger total number of Sibling and Spouse',
    0, 8
)

parch = st.slider(
    'Enter the Passenger total number of Parents and Child',
    0, 6
)

fare = st.number_input(
    'Enter the Fare of The Passenger'
)

embarked = st.selectbox(
    'Enter the Passenger Starting Station',
    ['Chebourg', 'Queenstown', 'Southampton']
)


user = {
    'Pclass': pclass,
    'Sex': sex,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

user = pd.DataFrame([user])

st.write(user)


# Load encoders

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot = pickle.load(file)

with open('data_sacler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Load model

model = load_model('model.h5')


# Label Encoding Sex

user['Sex'] = label.transform(user['Sex'])


# One Hot Encoding Embarked

embarked = onehot.transform(user[['Embarked']])

embarked = pd.DataFrame(
    embarked,
    columns=onehot.get_feature_names_out()
)


# Remove original Embarked

user = user.drop(columns=['Embarked'])


# Add encoded Embarked columns

user = pd.concat(
    [user, embarked],
    axis=1
)


# Scaling

nums = [
    'Pclass',
    'Sex',
    'SibSp',
    'Parch',
    'Fare'
]

user[nums] = scaler.transform(user[nums])


# Prediction

y = model.predict(user)

y = y[0][0]


# Function

def Chance(y):

    if y > 0.5:
        return 'The passenger will likely survive the Titanic Journey 🚢'
    else:
        return "The Passenger won't survive the Titanic Journey 💀"


# Button

if st.button('Predict Survival Chance'):

    st.write(
        'Probability of Passenger Survival Chance:',
        y
    )

    st.write(
        'Survival Percentage:',
        f'{y * 100:.2f}%'
    )

    st.write(
        Chance(y)      
    )
    