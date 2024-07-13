import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sample data to define preprocessors
data = {
    'Age': [25, 30, 35, 40, np.nan, 50],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', np.nan],
    'Education Level': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'PhD', 'Master'],
    'Job Title': ['Engineer', 'Scientist', 'Manager', 'Engineer', 'Manager', 'Scientist'],
    'Years of Experience': [3, 7, 10, np.nan, 20, 5],
    'Salary': [50000, 80000, 120000, 140000, 160000, 90000]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Dropping rows where target variable 'Salary' is NaN
df = df.dropna(subset=['Salary'])

# Features and target variable
X = df.drop('Salary', axis=1)
y = df['Salary']

# Preprocessing pipeline
numeric_features = ['Age', 'Years of Experience']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Gender', 'Education Level', 'Job Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Combine preprocessing and model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Fit the pipeline on the data
pipeline.fit(X, y)

# Streamlit app
st.title('Salary Prediction App')

st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 70, 25)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    education = st.sidebar.selectbox('Education Level', ('Bachelor', 'Master', 'PhD'))
    job = st.sidebar.selectbox('Job Title', ('Engineer', 'Scientist', 'Manager'))
    experience = st.sidebar.slider('Years of Experience', 0, 40, 5)
    data = {'Age': age,
            'Gender': gender,
            'Education Level': education,
            'Job Title': job,
            'Years of Experience': experience}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Predict
prediction = pipeline.predict(input_df)[0]
predicted_salary = f'${prediction:.2f}'

st.subheader('Predicted Salary')
st.write(predicted_salary)
