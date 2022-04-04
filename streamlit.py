from calendar import c
from pyrsistent import b
from sympy import get_indices
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
import folium
from PIL import Image 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

bank = pd.read_csv("bank.csv")
balanced = pd.read_csv("df_balance.csv")   

html_temp="""
<div style="background-color:#45B959;padding:1.5px">
<h1 style="color:while;text-align:center;">Artificial Intelligence Project</h1>
</div><br>
"""                    

st.markdown(html_temp, unsafe_allow_html=True)
st.text("Created by Akeef Aqil (1181101876), An'nissa Ariqah (11811001288) & Siti Nur Izzati (1181100895)")

st.info("Note: Please change the apperance to WIDE MODE under settings for better viewing.")

def main():
    section = ['Question 1','Question 2', 'Question 3']
    choice = st.sidebar.selectbox('Section', section)

    if choice == 'Question 1':
        st.markdown('# Vacation Planner')

        st.markdown("## The scenario will be:")
        st.markdown("Money on-hand = RM 10000")
        st.markdown("Vacation duration = 7 days")
        st.markdown("Hotel star rating = less than RM 400 per night")
        st.markdown("Tourist spots = 6 spots")
        st.markdown("One tourist spot = less than RM300")
        st.markdown("Food price = Less than RM 120 per meal")
        st.markdown("Transportation fees = less than RM 120 per trip")
        st.markdown("Transport frequency = 5 trip per day")

        st.markdown("## Presentation of problem domain in terms of GA")
        im = Image.open("2.jpeg")
        st.image(im, width=400, caption="problem domain")

        st.markdown("## Evolve function")
        im = Image.open("3 first.jpeg")
        st.image(im, width=600, caption="evolve1")

        im = Image.open("graph.jpeg")
        st.image(im, width=800, caption="evolve graph")
        
        im = Image.open("3 second.jpeg")
        st.image(im, width=600, caption="evolve2")

        st.markdown("## Best Individual for modification")
        corOption = st.selectbox("Select modification", ['Modification 1','Modification 2','Modification 3'])

        if corOption == 'Modification 1':
            im = Image.open("mod1.jpeg")
            st.image(im, width=700, caption="mod1")
        elif corOption == 'Modification 2':
            im = Image.open("mod2.jpeg")
            st.image(im, width=700, caption="mod2")
        else:
            im = Image.open("mod3.jpeg")
            st.image(im, width=700, caption="mod3")
       
    elif choice == 'Question 2':
        st.markdown('# Vaccine Distribution Modelling')

        q2 = pd.DataFrame({
        'State': ['ST-1','ST-2','ST-3','ST-4','ST-5'],
        'vaccine_Day': [5000, 10000, 7500, 8500, 9500], 
        'rentDay': [1700, 3400, 2950, 3000, 3900]
        })

        st.markdown("Maximum Vaccine Capacity Per Day")

        chart2 = alt.Chart(q2).mark_bar().encode(
            x='State',
            y='vaccine_Day',
            color=alt.Color('State', scale=alt.Scale(scheme='set1'))).properties(
            width=500,
            height=300
        ).interactive()
        chart2

        st.markdown("Total Rent Per Day")
        chart3 = alt.Chart(q2).mark_bar().encode(
            x='State',
            y='rentDay',
            color=alt.Color('State', scale=alt.Scale(scheme='set1'))).properties(
            width=500,
            height=300
        ).interactive()
        chart3
        
        st.markdown("## State")
        corOption = st.selectbox("Select STATE", ['ST-1','ST-2','ST-3','ST-4','ST-5'])

        if corOption == 'ST-1':
            im = Image.open("st1.png")
            st.image(im, width=700, caption="ST-1")
        elif corOption == 'ST-2':
            im = Image.open("st2.png")
            st.image(im, width=700, caption="ST-2")
        elif corOption == 'ST-3':
            im = Image.open("st3.png")
            st.image(im, width=700, caption="ST-3")
        elif corOption == 'ST-4':
            im = Image.open("st4.png")
            st.image(im, width=700, caption="ST-4")
        else:
            im = Image.open("st5.png")
            st.image(im, width=700, caption="ST-5")
    
    else:
        st.markdown('# Loan Application Modeling ')
              
        st.header("Classification: Naive Bayes")

        X = balanced.drop('Decision', axis=1)
        y = balanced['Decision']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        parameterNB = st.selectbox("Select Naive Bayes predictors", ["Option 1", "Option 2", "Option 3"])
        st.markdown("### Predictors: ")

        if parameterNB == "Option 1":
            st.markdown("Credit_Card_Exceed_Months: 1")
            st.markdown('Employment_Type: Employer') 
            st.markdown("Loan_Amount: RM456,509")
            st.markdown("Loan_Tenure_Year: 16")
            st.markdown('More_Than_One_Products: Yes')
            st.markdown("Credit_Card_More_Than_Months: 1")
            st.markdown("Number_of_Dependents: 2")
            st.markdown("Years_to_Financial_Freedom: 5")
            st.markdown("Number_of_Credit_Card_Facility: 2")
            st.markdown("Number_of_Properties: 2")
            st.markdown("Number_of_Bank_Products: 1")
            st.markdown("Number_of_Loan_to_Approve: 1")
            st.markdown("Property_Type: condominium")
            st.markdown("Years_for_Property_to_Completion: 10")
            st.markdown("State: Johor")
            st.markdown("Number_of_Side_Income: 2")
            st.markdown("Monthly_Salary: RM12,262")
            st.markdown("Total_Sum_of_Loan: RM853,141")
            st.markdown("Total_Income_for_Join_Application: RM16,025")
            st.markdown("Score: 7") 

            st.markdown("---"*10) 
            st.markdown('#### Decision is rejected')
            st.markdown("---"*10) 
        elif parameterNB == "Option 2":
            st.markdown("Credit_Card_Exceed_Months: 6")
            st.markdown('Employment_Type: Fresh graduate') 
            st.markdown("Loan_Amount: RM393,555")
            st.markdown("Loan_Tenure_Year: 14")
            st.markdown('More_Than_One_Products: Yes')
            st.markdown("Credit_Card_More_Than_Months: 2")
            st.markdown("Number_of_Dependents: 3")
            st.markdown("Years_to_Financial_Freedom: 15")
            st.markdown("Number_of_Credit_Card_Facility: 2")
            st.markdown("Number_of_Properties: 2")
            st.markdown("Number_of_Bank_Products: 5")
            st.markdown("Number_of_Loan_to_Approve: 2")
            st.markdown("Property_Type: terrace")
            st.markdown("Years_for_Property_to_Completion: 11")
            st.markdown("State: Kuala Lumpur")
            st.markdown("Number_of_Side_Income: 2")
            st.markdown("Monthly_Salary: RM5011")
            st.markdown("Total_Sum_of_Loan: RM673,748")
            st.markdown("Total_Income_for_Join_Application: RM13,892")
            st.markdown("Score: 7") 

            st.markdown("---"*10) 
            st.markdown('#### Decision is rejected')
            st.markdown("---"*10) 
        else:
            st.markdown("Credit_Card_Exceed_Months: 7")
            st.markdown('Employment_Type: Government') 
            st.markdown("Loan_Amount: RM601,606")
            st.markdown("Loan_Tenure_Year: 11")
            st.markdown('More_Than_One_Products: No')
            st.markdown("Credit_Card_More_Than_Months: 3")
            st.markdown("Number_of_Dependents: 4")
            st.markdown("Years_to_Financial_Freedom: 16")
            st.markdown("Number_of_Credit_Card_Facility: 4")
            st.markdown("Number_of_Properties: 4")
            st.markdown("Number_of_Bank_Products: 1")
            st.markdown("Number_of_Loan_to_Approve: 1")
            st.markdown("Property_Type: bungalow")
            st.markdown("Years_for_Property_to_Completion: 10")
            st.markdown("State: Penang")
            st.markdown("Number_of_Side_Income: 1")
            st.markdown("Monthly_Salary: RM5,124")
            st.markdown("Total_Sum_of_Loan: RM1,200,038")
            st.markdown("Total_Income_for_Join_Application: RM9,649")
            st.markdown("Score: 8")

            st.markdown("---"*10) 
            st.markdown('#### Decision is accepted')
            st.markdown("---"*10) 

        st.header("Classification: Decision Tree")

        parameterDT = st.selectbox("Select Decision Tree parameters", ["max_depth: 3 , min_samples_leaf: 2, criterion:gini", "max_depth: 6 , min_samples_leaf: 10, criterion:entropy", "Default - max_depth: none , min_samples_leaf: 1, criterion:gini","Hypertuned - max_depth: 20 , min_samples_leaf: 5, criterion:gini" ])
        
        if parameterDT == "max_depth: 3 , min_samples_leaf: 2, criterion:gini":
            im = Image.open("dt1EM.png")
            st.image(im, width=900, caption="Decision Tree 1")
        elif parameterDT == "max_depth: 6 , min_samples_leaf: 10, criterion:entropy":
            im = Image.open("dt2EM.png")
            st.image(im, width=900, caption="Decision Tree 2")
        elif parameterDT == "Default - max_depth: none , min_samples_leaf: 1, criterion:gini":
            im = Image.open("dtdefCM.png")
            st.image(im, width=900, caption="Decision Tree Default")
        else:
            im = Image.open("dthypertunedEM.png")
            st.image(im, width=900, caption="Decision Tree Hypertuned") 

        st.header("Cluster Analysis: K-Means")
        cluster = st.selectbox("Select k-mean cluster", ['3','4','6'])

        if cluster == '3':
            im = Image.open("cluster3.png")
            st.image(im, width=900, caption="cluster 3")
        elif cluster == '4':
            im = Image.open("cluster4.png")
            st.image(im, width=900, caption="cluster 4")
        else:
            im = Image.open("cluster6.png")
            st.image(im, width=900, caption="cluster 6")
    
if __name__ == '__main__':
    main()