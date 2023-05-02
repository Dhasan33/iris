# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)



rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(_model,sepal_length, sepal_width, petal_length, petal_width):
  species = _model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"
st.sidebar.title('Iris Flower Prediction APP')
s_len = st.sidebar.slider('Sepal Length',0.0,10.0)
s_wid = st.sidebar.slider('Sepal Width',0.0,10.0)
p_len = st.sidebar.slider('petal Length ',0.0,10.0)
p_wid = st.sidebar.slider('petal Width ',0.0,10.0)
clas = st.sidebar.selectbox('Classifier',('SVC','LogisticRegression','RandomForestClassifier')) 

if st.sidebar.button('Predict'):
	if clas == 'SVC':
		s = prediction(svc_model,s_len,s_wid,p_len,p_wid)
		st.write('The flower of the given identifications is :',s)
		st.write('The accuracy of the model is :',svc_model.score(X_train, y_train))
	elif clas == 'LogisticRegression':	
		s = prediction(log_reg,s_len,s_wid,p_len,p_wid)
		st.write('The flower of the given identifications is :',s)
		st.write('The accuracy of the model is :',log_reg.score(X_train, y_train))
	else:		
		s = prediction(rf_clf,s_len,s_wid,p_len,p_wid)
		st.write('The flower of the given identifications is :',s)
		st.write('The accuracy of the model is :',rf_clf.score(X_train, y_train))			
