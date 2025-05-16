import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Tamil Nadu Traffic Accident Analysis",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embedded dataset
def load_data():
    data = {
        'District': [
            'Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore',
            'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kancheepuram',
            'Karur', 'Krishnagiri', 'Madurai', 'Nagapattinam', 'Namakkal',
            'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Ranipet',
            'Salem', 'Sivaganga', 'Tenkasi', 'Thanjavur', 'Theni',
            'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tirupathur', 'Tiruppur',
            'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram',
            'Virudhunagar', 'Mayiladuthurai'
        ],
        'Latitude': [
            10.059970653660493, 13.22892868525454, 12.025966679962728, 11.2926216630837, 8.8581025224334,
            8.857969861849114, 8.319459866925097, 12.763968801762143, 11.30613256458765, 11.89439917787825,
            8.113214718626914, 13.334504186890968, 12.578434524402319, 9.167865108730519, 9.000037319639054,
            9.008724804193886, 9.673332336277458, 10.886160373977308, 10.375697602531638, 9.60176027108923,
            11.365190920973088, 8.76721623358623, 9.606795566943699, 10.014990138115305, 10.508384913193698,
            12.318467787661575, 9.098205801870979, 10.828289411274863, 11.258280128741234, 8.255477269959988,
            11.341496685457912, 8.937882680280104, 8.357783761419038, 13.218870454893333, 13.310976181910076,
            12.446185414640535, 9.675375730453538
        ],
        'Longitude': [
            76.89068845602553, 79.23693210604863, 78.2606099749584, 76.98815293937912, 78.48070764044508,
            76.63755408446087, 80.13728160831513, 77.53511992640007, 79.15008913741593, 77.74684430435764,
            78.58027208471124, 78.68684111737312, 77.23941782210211, 80.37833851105823, 79.60053129344446,
            80.25799576625676, 80.07930940171059, 78.89159991524434, 80.18749694009247, 76.85397000820768,
            77.28393144967659, 76.68090915564215, 77.80132132305306, 78.05470915875793, 77.58539612709558,
            79.81495003660771, 77.92701330677436, 77.62373803874952, 78.67078433263299, 77.06369689989906,
            79.70878792301616, 76.79820257471908, 80.44754774640207, 79.58897907718664, 77.29486272613669,
            76.52208846849442, 79.76184571381934
        ],
        'Accidents_2019': [
            1192, 991, 2298, 1018, 888,
            978, 2396, 2113, 830, 1617,
            640, 1651, 734, 800, 2463,
            1379, 1529, 1656, 1671, 1102,
            1006, 2197, 2351, 1404, 698,
            1283, 2495, 1325, 2170, 1560,
            2362, 1212, 2085, 1242, 2392,
            1628, 1841
        ],
        'Accidents_2020': [
            1179, 965, 2290, 940, 874,
            889, 2355, 2037, 780, 1555,
            545, 1600, 639, 797, 2370,
            1357, 1515, 1614, 1643, 1067,
            994, 2166, 2281, 1346, 613,
            1256, 2430, 1284, 2126, 1499,
            2306, 1207, 2058, 1215, 2349,
            1545, 1812
        ],
        'Population': [
            80.1, 32.6, 15.4, 35.3, 82.1,
            28.1, 60.1, 5.0, 35.0, 30.9,
            19.0, 50.4, 46.2, 63.9, 27.9,
            25.8, 19.3, 23.6, 52.4, 39.3,
            10.5, 26.6, 26.0, 64.2, 65.5,
            17.6, 89.8, 27.7, 88.0, 39.9,
            7.8, 34.3, 58.9, 62.9, 50.1,
            43.1, 52.0
        ]
    }
    
    tn_df = pd.DataFrame(data)
    tn_df['Population_Lakhs'] = tn_df['Population'] / 10
    tn_df['Accident_Reduction'] = tn_df['Accidents_2019'] - tn_df['Accidents_2020']
    tn_df['Accident_Rate_2020'] = tn_df['Accidents_2020'] / tn_df['Population_Lakhs']
    return tn_df

tn_df = load_data()

# Sidebar navigation
st.sidebar.title("Analysis Modules")
analysis_type = st.sidebar.radio("Select Analysis", 
                                ["District Overview", 
                                 "Accident Prediction Model",
                                 "Geospatial Analysis"])

# Main app
st.title("Tamil Nadu Road Accident Analysis System")


if analysis_type == "District Overview":
    st.header("District-Level Accident Analysis")
    
    # Top districts analysis
    st.subheader("Top Districts by Accident Volume (2020)")
    top_n = st.slider("Select number of districts to display", 5, 15, 10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(tn_df.nlargest(top_n, 'Accidents_2020'),
                     x='District',
                     y='Accidents_2020',
                     color='Accidents_2020',
                     title=f"Top {top_n} Districts by Accidents (2020)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(tn_df.nlargest(top_n, 'Accidents_2020'),
                     values='Accidents_2020',
                     names='District',
                     title=f"Accident Distribution - Top {top_n} Districts")
        st.plotly_chart(fig, use_container_width=True)
    
    # Year comparison
    st.subheader("Year-over-Year Comparison (2019 vs 2020)")
    try:
        fig = px.scatter(tn_df,
                        x='Accidents_2019',
                        y='Accidents_2020',
                        color='District',
                        size='Population',
                        hover_name='District',
                        trendline="ols",
                        title="2019 vs 2020 Accidents with Population Size")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Trendline disabled: {str(e)}")
        fig = px.scatter(tn_df,
                        x='Accidents_2019',
                        y='Accidents_2020',
                        color='District',
                        size='Population',
                        hover_name='District',
                        title="2019 vs 2020 Accidents (trendline disabled)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Accident reduction analysis
    st.subheader("Districts with Highest Accident Reduction")
    fig = px.bar(tn_df.nlargest(10, 'Accident_Reduction'),
                 x='District',
                 y='Accident_Reduction',
                 color='Accident_Reduction',
                 title="Top 10 Districts Showing Accident Reduction (2019-2020)")
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Accident Prediction Model":
    st.header("Accident Risk Prediction Model")
    
    # Model development
    st.subheader("Random Forest Regression Model")
    
    # Prepare data
    X = tn_df[['Accidents_2019', 'Population']]
    y = tn_df['Accidents_2020']
    
    # Train model
    model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=5)
    model.fit(X, y)
    tn_df['Predicted_2020'] = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, tn_df['Predicted_2020'])
    r2 = r2_score(y, tn_df['Predicted_2020'])
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R-squared Score", f"{r2:.2f}")
    
    # Show actual vs predicted
    st.subheader("Model Performance: Actual vs Predicted")
    try:
        fig = px.scatter(tn_df,
                        x='Accidents_2020',
                        y='Predicted_2020',
                        hover_name='District',
                        trendline="ols",
                        title="Actual vs Predicted Accidents (2020)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Trendline disabled: {str(e)}")
        fig = px.scatter(tn_df,
                        x='Accidents_2020',
                        y='Predicted_2020',
                        hover_name='District',
                        title="Actual vs Predicted Accidents (trendline disabled)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance,
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 title="Feature Importance in Accident Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk categorization
    st.subheader("District Risk Categorization")
    tn_df['Risk_Level'] = pd.qcut(tn_df['Predicted_2020'], 5,
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    fig = px.bar(tn_df.sort_values('Predicted_2020'),
                 x='District',
                 y='Predicted_2020',
                 color='Risk_Level',
                 title="Predicted Accident Risk by District",
                 category_orders={"Risk_Level": ["Very Low", "Low", "Medium", "High", "Very High"]})
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Geospatial Analysis":
    st.header("Geospatial Accident Distribution")
    
    # Map visualization
    st.subheader("Accident Hotspots (2020)")
    fig = px.scatter_mapbox(tn_df,
                          lat="Latitude",
                          lon="Longitude",
                          size="Accidents_2020",
                          color="Accidents_2020",
                          hover_name="District",
                          hover_data=["Accidents_2019", "Accidents_2020", "Population"],
                          zoom=6,
                          height=600,
                          mapbox_style="open-street-map",
                          title="Geographical Distribution of Accidents")
    st.plotly_chart(fig, use_container_width=True)
    
    # Choropleth-like visualization
    st.subheader("Accident Rate per Lakh Population")
    fig = px.scatter_mapbox(tn_df,
                          lat="Latitude",
                          lon="Longitude",
                          size="Accident_Rate_2020",
                          color="Accident_Rate_2020",
                          hover_name="District",
                          hover_data=["Accident_Rate_2020", "Population"],
                          zoom=6,
                          height=600,
                          mapbox_style="carto-positron",
                          title="Accident Rate per Lakh Population")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
