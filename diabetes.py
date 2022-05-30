import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.header("Diabetes Detection using machine learning ")
#image=Image.open('/Users/download.png')
#st.image(image, caption='plot', use_column_width=True)
df = pd.read_csv('/Users/diabetes.csv')
#st.subheader('Data information:')
#st.dataframe(df)
#st.write(df.describe())
#chart = st.bar_chart(df)

X= df.iloc[:, 0:8].values
Y= df.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies (Number of times pregnant)', 0, 17, 3)
    glucose= st.sidebar.slider('glucose (Plasma glucose concentration a 2 hours in an oral glucose tolerance test)',0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure (Diastolic blood pressure (mm Hg))', 0, 122, 72)
    skin_thickness= st.sidebar.slider('skin_thickness (Triceps skin fold thickness (mm))', 0, 99, 23)
    insulin= st.sidebar.slider('insulin (2-Hour serum insulin (mu U/ml))', 0.0, 846.0, 30.0)
    BMI= st.sidebar.slider('BMI (weight in kg/(height in m)^2)',0.0,  67.1, 32.0)
    DPF= st.sidebar.slider('DPF (a function which scores likelihood of diabetes based on family history)(<0.5=no history, >1.5=has history)', 0.078, 2.42, 0.3725)
    Age= st.sidebar.slider('age', 21, 81, 29)
 
    user_data = {'pregnencies': pregnancies,
    'glucose': glucose,
    'blood_pressure': blood_pressure,
    'skin_thickness': skin_thickness,
    'insulin': insulin,
    'BMI': BMI,
    'DPF': DPF,
    'age': Age
    }
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('classification:')
#st.write(prediction)
if (prediction[0] == 0):
  st.write('The person is not diabetic')
else:
  st.write('The person is diabetic')
