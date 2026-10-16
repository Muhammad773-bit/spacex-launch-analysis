import streamlit as st
import pandas as pd
import folium
import joblib
from streamlit_folium import st_folium

# Load data and model
launch_data = pd.read_csv('spacex_launches_preprocessed.csv')
model = joblib.load('launch_success_model.pkl')

# Launch sites for map
launch_sites = {
    '5e9e4502f509094188566f88': {'name': 'Cape Canaveral SLC-40', 'lat': 28.5623, 'lon': -80.5774},
    '5e9e4502f509092b78566f87': {'name': 'Vandenberg SLC-4E', 'lat': 34.6321, 'lon': -120.6107},
    '5e9e4501f509094ba4566f84': {'name': 'Kennedy LC-39A', 'lat': 28.6082, 'lon': -80.6041}
}

# Streamlit app
st.title("SpaceX Launch Analysis & Prediction Platform")

# Sidebar for filters
st.sidebar.header("Filter Data")
year_filter = st.sidebar.slider("Select Year", int(launch_data['year'].min()), int(launch_data['year'].max()), (2006, 2025))
site_filter = st.sidebar.multiselect("Select Launch Site", launch_data['launchpad'].unique(), default=launch_data['launchpad'].unique())

# Filter data
filtered_data = launch_data[(launch_data['year'].between(year_filter[0], year_filter[1])) & (launch_data['launchpad'].isin(site_filter))]

# Historical data
st.header("Historical Launch Data")
st.dataframe(filtered_data[['flight_number', 'name', 'year', 'success']])

# Geospatial map
st.header("Launch Sites Map")
m = folium.Map(location=[28.5, -80.5], zoom_start=5)
for site_id, site_info in launch_sites.items():
    if site_id in filtered_data['launchpad'].values:
        site_data = filtered_data[filtered_data['launchpad'] == site_id]
        success_count = site_data['success'].sum()
        total_count = len(site_data)
        popup_text = f"{site_info['name']}<br>Successes: {success_count}/{total_count}"
        folium.Marker(
            location=[site_info['lat'], site_info['lon']],
            popup=popup_text,
            icon=folium.Icon(color='blue' if success_count/total_count > 0.8 else 'red')
        ).add_to(m)
st_folium(m, width=700, height=500)

# Prediction tool
st.header("Predict Launch Success")
year = st.number_input("Year", min_value=2006, max_value=2030, value=2025)
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)
precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0)
rocket = st.selectbox("Rocket Type", launch_data['rocket'].unique())
launchpad = st.selectbox("Launch Site", launch_data['launchpad'].unique())

# Encode inputs
rocket_encoded = pd.Series([rocket]).map(lambda x: launch_data[launch_data['rocket'] == x]['rocket_encoded'].iloc[0])
launchpad_encoded = pd.Series([launchpad]).map(lambda x: launch_data[launch_data['launchpad'] == x]['launchpad_encoded'].iloc[0])

# Predict
input_data = pd.DataFrame({
    'year': [year],
    'weather_temp': [temp],
    'weather_wind_speed': [wind],
    'weather_precipitation': [precip],
    'rocket_encoded': [rocket_encoded.iloc[0]],
    'launchpad_encoded': [launchpad_encoded.iloc[0]]
})
if st.button("Predict"):
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f"Launch Success Probability: {prediction:.2%}")