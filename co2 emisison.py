import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score

# Load the dataset
def load_data():
    fname = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
    df = pd.read_csv(fname)
    return df

df = load_data()

# Load the pre-trained model
with open('C:\\Users\\deeks\\OneDrive\\pj\\model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare dataset for evaluation
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Evaluate the model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = model.predict(test_x)

# Function to plot results
def plot_results():
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', label='Train Data')
    plt.plot(train_x, model.coef_[0][0]*train_x + model.intercept_[0], 'red', label='Fitted Line')
    plt.xlabel("Engine Size")
    plt.ylabel("Emission")
    plt.title("Engine Size vs CO2 Emissions")
    plt.legend()
    st.pyplot(fig)

# Function to predict CO2 emissions
def predict_emission(engine_size):
    return model.predict(np.array([[engine_size]]))[0][0]

# Streamlit App
st.title("Fuel Consumption CO2 Emissions Prediction")
st.write("This app predicts CO2 emissions based on engine size using a pre-trained model.")

# Input
engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=8.4, step=0.1, value=3.0)

# Prediction
predicted_emission = predict_emission(engine_size)
st.write(f"Predicted CO2 Emission: {predicted_emission:.2f} g/km")

# Display results
# st.subheader("Train Data Plot")
# plot_results()

# Model Evaluation
st.subheader("Model Evaluation Metrics")
st.write(f"Mean Absolute Error: {np.mean(np.absolute(test_y_hat - test_y)):.2f}")
st.write(f"Residual Sum of Squares (MSE): {np.mean((test_y_hat - test_y) ** 2):.2f}")
st.write(f"R2 Score: {r2_score(test_y_hat, test_y):.2f}")