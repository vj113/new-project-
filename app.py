import streamlit as st
#from textblob import textblob
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


s = pd.read_csv("social_media_usage.csv",)

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


def gender(x):
    x = np.where(x == 2, 1, 0)
    return x


ss = pd.DataFrame({
    "Linkedin":s["web1h"].apply(clean_sm),
    "income":np.where(s["income"]>9,np.nan, s["income"]),
    "education":np.where(s["educ2"]>8,np.nan,s["educ2"]),
    "parents":np.where(s["par"]== 1,1,0),
    "married":np.where(s["marital"]==1,1,0),
    "female":np.where(s["gender"]==2,1,0),
    "age": np.where(s["age"]>98, np.nan,s["age"])})

ss=ss.dropna()


y = ss["Linkedin"]
X = ss[['income', 'education', 'parents', 'married', 'female', 'age']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987)


lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


import streamlit as st

st.title("Linkedin Users Prediction Application")
st.markdown("Please configure demongrapchis to predict if someone will likely use Linkedin")

income= st.selectbox("Income:", 
                      options= ["less than $10,000",
                                "10 to under $20,000",
                                "20 to under $30,000",
                                "30 to under $40,000", 
                                "40 to under $50,000", 
                                "50 to under $75,000",
                                "75 to under $100,000",
                                "100 to under $150,000",
                                "150,00 or more"])





education= st.selectbox("Education:",
                        options = ["Less than high school", 
                                   "High school incomplete", 
                                   "High school graduate ",
                                   "Some college, no degree",
                                   "Two-year associate degree from a college or university",
                                   "Four-year college or university degree/Bachelors degree ",
                                   "Some postgraduate or professional schooling, no postgraduate degree"])
                
 


parents= st.radio("Enter Parental Status", ["Not a parent", "Parent"])

married= st.radio("Enter Marital Status",["Married", "Not Married"])

female= st.radio("Enter Gender",["Female", "Male"])

age= st.slider(label="Enter Age",
               min_value=18,
               max_value=100,
               value=60)



person= [8,7,0,1,1,42]

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])
probability=(round(probs[0][1],2)*100)


if predicted_class == 1:
    predicted_class1 = "Is a Linkedin user"
else:
    predicted_class1= "Is not a Linkedin userd"


st.write(f"It is predicted that this person {predicted_class}")
st.write(f"The probability that this person is a user is {probability}")

