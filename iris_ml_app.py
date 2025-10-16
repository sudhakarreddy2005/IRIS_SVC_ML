import streamlit as st
import numpy as np
import joblib

sv_model = joblib.load('joblib_iris_ml.joblib')


st.title("🌸 Iris Flower Classification App")
st.write("This web app predicts the **Iris flower species** based on user input features using a trained SVM model.")


st.sidebar.header("Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 4.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.3)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return features

input_data = user_input_features()

prediction = sv_model.predict(input_data)
prediction_proba = None
try:
    prediction_proba = sv_model.predict_proba(input_data)
except:
    pass  # some models may not support predict_proba

# Mapping class labels
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
st.subheader("Prediction Result:")
st.success(f"🌼 The predicted species is: **{class_names[prediction[0]]}**")

# Show probability if available
if prediction_proba is not None:
    st.subheader("Prediction Probability:")
    st.bar_chart(prediction_proba[0])

# Footer
st.markdown("---")
st.caption("Developed by REDDY • Powered by  SVM Model")
