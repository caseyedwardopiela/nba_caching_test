import streamlit as st
import pandas as pd
#import base64
#import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=nF-PQj0k5-o&ab_channel=PythonTutorialsforDigitalHumanities
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.set_page_config(
  page_title="Cache"
)

st.sidebar.success("Select a page above to continue...")

st.title('2022-2023 NBA Cache')

st.markdown("""
Cache test
""")

allstar_data = pd.read_csv('raw_data_for_allstars.csv').drop(['Unnamed: 0'], axis = 1)

training_df = allstar_data[allstar_data['Season'] < 2023]

def load_models(training):
    training_columns = list(training.columns)[3:9]
    output_column = 'Allstar'
    y = list(training[output_column])
    x = training.loc[:,training_columns]
    rf_allstar = RandomForestClassifier()
    rf_allstar.fit(x, y)
    xgb_allstar = xgb.XGBClassifier()
    xgb_allstar.fit(x, y)
    logistic_allstar = LogisticRegression()
    logistic_allstar.fit(x, y)
    return(rf_allstar, xgb_allstar, logistic_allstar)

rf, xg, lr = load_models(training_df)

def all_star_model(rf, xg, lr, p, rb, a, s, b, rec):

  df_temp = pd.DataFrame({'Points': [p], 'Rebounds': [rb], 'Assists': [a], 'Steals': [s], 'Blocks': [b], 'Record': [rec]})

  # Random Forest
  rf_allstar_results = rf.predict_proba(df_temp)
  rf_allstar_results = [i[1] for i in rf_allstar_results][0]

  # XGBoost
  xgb_allstar_results = xg.predict_proba(df_temp)
  xgb_allstar_results = [i[1] for i in xgb_allstar_results][0]

  # Logistic
  logistic_allstar_results = lr.predict_proba(df_temp)
  logistic_allstar_results = [i[1] for i in logistic_allstar_results][0]

  allstar_prediction = round((rf_allstar_results + xgb_allstar_results + logistic_allstar_results) / 3, 3)
  return allstar_prediction


with st.form(key ='Form1'):
    with st.sidebar:
        points = st.slider('Points Per Game', min_value = float(0.0), max_value = float(50.0), step = float(0.1))
        rebounds = st.slider('Rebounds Per Game', min_value = float(0.0), max_value = float(20.0), step = float(0.1))
        assists = st.slider('Assists Per Game', min_value = float(0.0), max_value = float(20.0), step = float(0.1))     
        steals = st.slider('Steals Per Game', min_value = float(0.0), max_value = float(5.0), step = float(0.1))
        blocks = st.slider('Blocks Per Game', min_value = float(0.0), max_value = float(5.0), step = float(0.1))
        record = st.slider('Team Record', min_value = float(0.0), max_value = float(1), step = float(0.05))
        submitted = st.form_submit_button(label = 'Determine All-Star Likelihood')
        
probability = all_star_model(training_df, points, rebounds, assists, steals, blocks, record) 
st.write('The likelihood of your player making the NBA All-Star Game Roster is: ', probability)
