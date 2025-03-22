#!/usr/bin/env python
# coding: utf-8

# # Setup and Get Race Results with the Ergast API

# In[1]:



# In[2]:


import requests
import pandas as pd
from time import sleep

def get_race_results(year):
    url = f"http://ergast.com/api/f1/{year}/results.json?limit=1000"
    response = requests.get(url)
    data = response.json()
    
    races = data['MRData']['RaceTable']['Races']
    rows = []

    for race in races:
        race_name = race['raceName']
        round_num = race['round']
        circuit = race['Circuit']['circuitName']
        date = race['date']

        for result in race['Results']:
            position = result['position']
            driver = result['Driver']
            constructor = result['Constructor']

            row = {
                'year': year,
                'round': int(round_num),
                'race': race_name,
                'circuit': circuit,
                'date': date,
                'driver_id': driver['driverId'],
                'driver_name': f"{driver['givenName']} {driver['familyName']}",
                'constructor': constructor['name'],
                'grid': int(result['grid']),
                'position': int(position),
                'status': result['status']
            }

            rows.append(row)
    
    return rows

# Fetch results from 2018–2024
all_results = []
for year in range(2018, 2025):  # updated to include 2024
    print(f"Fetching results for {year}...")
    all_results.extend(get_race_results(year))
    sleep(1)

df_results = pd.DataFrame(all_results)
df_results.dropna(inplace=True)
df_results = df_results[df_results['status'] == 'Finished']
df_results['race_date'] = pd.to_datetime(df_results['date'])
df_results.sort_values(by=['race_date'], inplace=True)
df_results['grid'] = df_results['grid'].astype(int)
df_results['position'] = df_results['position'].astype(int)

# Feature Engineering
df_results['driver_avg_finish'] = df_results.groupby('driver_id')['position'].expanding().mean().shift().reset_index(level=0, drop=True)
df_results['driver_win_ratio'] = df_results.groupby('driver_id')['position'].apply(lambda x: (x == 1).expanding().mean().shift()).reset_index(level=0, drop=True)
points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
df_results['points'] = df_results['position'].map(points_system).fillna(0)
df_results['constructor_avg_points'] = df_results.groupby('constructor')['points'].expanding().mean().shift().reset_index(level=0, drop=True)
df_results['driver_circuit_avg_finish'] = df_results.groupby(['driver_id', 'circuit'])['position'].expanding().mean().shift().reset_index(level=[0,1], drop=True)
df_results.fillna(0, inplace=True)

# Add target
df_results['target'] = (df_results['position'] == 1).astype(int)



# # Data Preparation and Feature Engineering

# In[3]:


# Drop rows with missing values
df_results.dropna(inplace=True)

# Convert data types
df_results['grid'] = df_results['grid'].astype(int)
df_results['position'] = df_results['position'].astype(int)

# Filter out non-finishers (e.g., 'Retired', 'Disqualified')
df_results = df_results[df_results['status'] == 'Finished']


# In[4]:


# Calculate cumulative statistics up to each race
df_results['race_date'] = pd.to_datetime(df_results['date'])
df_results.sort_values(by=['race_date'], inplace=True)

# Driver's average finishing position and win ratio up to each race
df_results['driver_avg_finish'] = df_results.groupby('driver_id')['position'].expanding().mean().shift().reset_index(level=0, drop=True)
df_results['driver_win_ratio'] = df_results.groupby('driver_id')['position'].apply(lambda x: (x == 1).expanding().mean().shift()).reset_index(level=0, drop=True)

# Constructor's average points up to each race
# Assuming a points system where 1st=25, 2nd=18, ..., 10th=1
points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
df_results['points'] = df_results['position'].map(points_system).fillna(0)
df_results['constructor_avg_points'] = df_results.groupby('constructor')['points'].expanding().mean().shift().reset_index(level=0, drop=True)

# Driver's performance at specific circuits
df_results['driver_circuit_avg_finish'] = df_results.groupby(['driver_id', 'circuit'])['position'].expanding().mean().shift().reset_index(level=[0,1], drop=True)

# Fill NaN values resulting from the shift operation
df_results.fillna(0, inplace=True)


# # Modeling

# In[5]:


# Target variable: 1 if the driver won the race, else 0
df_results['target'] = (df_results['position'] == 1).astype(int)

# Features for modeling
features = [
    'grid',
    'driver_avg_finish',
    'driver_win_ratio',
    'constructor_avg_points',
    'driver_circuit_avg_finish'
]

X = df_results[features]
y = df_results['target']


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))


# # Making Predictions for the Current Season

# In[7]:


# Example: Predicting for the Australian Grand Prix 2025

# Latest data (this should be updated with real data)
latest_data = {
    'driver_id': 'max_verstappen',
    'grid': 1,
    'driver_avg_finish': 2.0,
    'driver_win_ratio': 0.35,
    'constructor_avg_points': 18.5,
    'driver_circuit_avg_finish': 1.5
}

# Convert to DataFrame
latest_df = pd.DataFrame([latest_data])

# Predict probability of winning
win_probability = model.predict_proba(latest_df[features])[:, 1]
print(f"Predicted probability of winning: {win_probability[0]:.2f}")


# # Building the Dashboard

# In[8]:




# In[12]:


import streamlit as st
import pandas as pd
import numpy as np

st.title("🏁 F1 Race Winner Predictor")

st.sidebar.header("Race Info")
circuit = st.sidebar.selectbox("Select Circuit", ["Shanghai", "Monaco", "Silverstone"])
weather = st.sidebar.selectbox("Weather", ["Dry", "Wet", "Mixed"])

st.sidebar.header("Driver Stats")
driver = st.sidebar.selectbox("Select Driver", ["Max Verstappen", "Charles Leclerc", "Lando Norris", "Lewis Hamilton"])
qualifying_pos = st.sidebar.slider("Qualifying Position", 1, 20, 1)
avg_finish = st.sidebar.slider("Average Finish This Season", 1.0, 20.0, 5.0)
win_ratio = st.sidebar.slider("Season Win Ratio", 0.0, 1.0, 0.2)

# Dummy prediction logic
score = (21 - qualifying_pos) * 0.4 + (1 / avg_finish) * 10 + win_ratio * 100
prob_win = min(score / 100, 1.0)

st.subheader("📊 Prediction")
st.markdown(f"**Driver:** {driver}")
st.markdown(f"**Circuit:** {circuit}")
st.markdown(f"**Predicted Probability of Winning:** `{prob_win:.2%}`")

if prob_win > 0.8:
    st.success("🏆 High chance of winning!")
elif prob_win > 0.5:
    st.info("⚠️ Possible podium finish")
else:
    st.warning("📉 Low chance of winning")

 


# In[ ]:




