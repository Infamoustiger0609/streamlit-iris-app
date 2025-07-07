import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
model = joblib.load('iris_model.pkl')

# App title
st.title("🌸 Iris Flower Species Prediction")
st.write("Enter flower measurements to predict its species.")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)',
                                       'petal length (cm)', 'petal width (cm)'])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    iris = load_iris()
    st.success(f"🌼 Predicted species: {iris.target_names[prediction].capitalize()}")

    # Visualization
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(iris.target_names, prediction_proba[0], color=['#FF9999','#66B3FF','#99FF99'])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
