import streamlit as st
import numpy as np
import pandas as pd
from pickle import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
import warnings 
warnings.filterwarnings("ignore")
# Loading pretrained classifiers from pickle file
df =pd.read_csv(r'C:\Users\Aishwarya\Desktop\Prassu\prasanna\Datasets\credit_risk_dataset.csv')

scaler = load(open(r'C:\Users\Aishwarya\Desktop\Prassu\prasanna\Data science cource\internship\models\standard_scaler.pkl','rb'))
ohe =load(open(r'C:\Users\Aishwarya\Desktop\Prassu\prasanna\Data science cource\internship\models\one_hot_encoding.pkl','rb'))
loan_grade_encoder =load(open(r'C:\Users\Aishwarya\Desktop\Prassu\prasanna\Data science cource\internship\models\grade.pkl','rb'))

#knn_classifier = load(open('models/knn_model.pkl', 'rb'))
#lr_classifier = load(open('models/lr_model.pkl', 'rb'))
#dt_classifier = load(open('models/dt_model.pkl', 'rb'))
#sv_classifier = load(open('models/sv_model.pkl', 'rb'))
rf_classifier = load(open(r'C:\Users\Aishwarya\Desktop\Prassu\prasanna\Data science cource\internship\models\rf_model.pkl', 'rb'))

## Taking Input From User
age = st.selectbox("Select the Processor:- ", df["person_age"].unique())
income=st.selectbox("Enter Income",df['person_income'].unique())
emp_length=st.selectbox("Enter Person Employee Length",df['person_emp_length'].unique())
loan_amt=st.selectbox("Enter Loan Amount",df['loan_amnt'].unique())
loan_int_rate=st.selectbox("Enter Loan Interest Rate",df['loan_int_rate'].unique())
loan_percent_income=st.selectbox("Enter Loan Percent Income:",df['loan_percent_income'].unique())
cb_person_cred_hist_length=st.selectbox("Enter Credit History Length:",df['cb_person_cred_hist_length'].unique())

person_home_ownership=st.selectbox("Enter Person Home Ownership:",df['person_home_ownership'].unique())
loan_intent=st.selectbox("Enter Loan Intent:",df['loan_intent'].unique())
loan_grade=st.selectbox("Enter Loan Grade:",df['loan_grade'].unique())
cb_person_default_on_file=st.selectbox(" Enter Historic Default:",df['cb_person_default_on_file'].unique())

# age = st.number_input("Select the Processor:- ",format="%.2f")
# income=st.number_input(" Enter Income",format="%.2f")
# emp_length=st.number_input("Enter Person Employee Length",format="%.2f")
# loan_amt=st.number_input("Enter Loan Amount",format="%.2f")
# loan_int_rate=st.number_input("Enter Loan Interest Rate",format="%.2f")
# loan_percent_income=st.number_input("Enter Loan Percent Income:",format="%.2f")
# cb_person_cred_hist_length=st.number_input("Enter Credit History Length:",format="%.2f")

# person_home_ownership=st.text_input("Enter Person Home Ownership:")
# loan_intent=st.text_input("Enter Loan Intent:")
# loan_grade=st.text_input("Enter Loan Grade:")
# cb_person_default_on_file=st.text_input(" Enter Historic Default:")

btn_click = st.button("Predict")

query_point = np.array([age,income,emp_length,loan_amt,loan_int_rate,loan_percent_income,cb_person_cred_hist_length]).reshape(1, -1)
query_point_transformed = scaler.transform(query_point)

query_point_ohe=np.array([person_home_ownership,loan_intent,
       cb_person_default_on_file]).reshape(1,-1)

query_point_transformed_ohe = ohe.transform(query_point_ohe).reshape(1,-1)

loan_grade_transformed=0
for i in loan_grade_encoder:
    if i in loan_grade:
        #print(loan_grade,loan_grade_encoder[i])
        loan_grade_transformed=loan_grade_encoder[i]

query_point_lb = np.array([loan_grade_transformed]).reshape(1,-1)


## Prediction
if btn_click == True:
    if age and income and emp_length and loan_amt and loan_int_rate and loan_percent_income and cb_person_cred_hist_length and person_home_ownership and loan_intent and loan_grade and cb_person_default_on_file:
        new_query_point=np.append(query_point,query_point_transformed_ohe).reshape(1,-1)
        new_query_point1=np.append(new_query_point,query_point_lb).reshape(1,-1)
        
        pred = rf_classifier.predict(new_query_point1)
        if pred==0:
            st.success("Non Default")
            
        else:
            st.success("Default")
            
        
    else:
        st.error("Enter the values properly.")