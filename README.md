# 🚨 Tamil Nadu Traffic Accident Analysis

A Streamlit-based interactive dashboard to analyze, visualize, and predict road accidents across districts in Tamil Nadu. This project helps identify accident hotspots, compare trends, and assess risk using a machine learning model.

🔗 **Live App**: [traffic-accident-analysis-model.streamlit.app](https://traffic-accident-analysis-model.streamlit.app/)

---

## 📌 Features

- 📍 **District Overview**: Bar, pie, and scatter plots comparing accident statistics across districts (2019 vs 2020)
- 🔮 **Accident Prediction Model**: Random Forest regression to forecast accident counts and assign risk levels
- 🗺️ **Geospatial Analysis**: Interactive Mapbox maps showing accident density and rate per lakh population

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly Express
- **Backend**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest Regressor)
- **Mapping**: Plotly's Mapbox

---

## 🚀 How to Run Locally
#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/traffic-accident-analysis-model.git
cd traffic-accident-analysis-model
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Launch the Application

```bash
streamlit run traffic_app.py
```
