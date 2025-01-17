import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import traceback

model = tf.keras.models.load_model("./model_files/model.h5")

try:
    with open("./model_files/ohe_geo_gen.pkl", "rb") as file:
        ohe = pickle.load(file)
except Exception as e:
    print("error in reading pickle files: ", e)
    print(traceback.format_exc())

try:
    with open("./model_files/ssclaer.pkl", "rb") as file:
        sscaler = pickle.load(file)
except Exception as e:
    print("error in reading pickle files: ", e)
    print(traceback.format_exc())

######################### Title ###########################
st.title("Customer Churn Prediction")

######## User Input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox("Gender", ohe.categories_[1])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_cards = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'creditscore': [credit_score],
    'geography': [geography],
    'gender': [gender],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'numofproducts': [num_of_products],
    'hascrcard': [has_cr_cards],
    'isactivemember': [is_active_member],
    'estimatedsalary': [estimated_salary]
}


data = pd.DataFrame(input_data, index = ['a']).reset_index(drop=True)

encoded = ohe.transform(data[['geography', 'gender']])

columns = ohe.get_feature_names_out(['geography', 'gender'])
encoded_df = pd.DataFrame(encoded, columns=columns).reset_index(drop=True)


data.drop(['geography',	'gender'], axis = 1, inplace = True)
data = pd.concat([data, encoded_df], axis = 1)

x_test = pd.DataFrame(sscaler.transform(data), columns = data.columns)

predicted_val = model.predict(x_test)
predict_proba = predicted_val[0][0]

print(predict_proba)
st.write(predict_proba)

if predict_proba >= 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
