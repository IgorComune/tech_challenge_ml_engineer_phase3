# Tech Challenge ML Engineer - Phase 3
## Amazon Delivery Time Prediction

This project was developed as part of Phase 3 Tech Challenge, focusing on delivery time prediction (ETA) using Amazon Delivery data. The goal is to create a complete Machine Learning model with data pipeline, API, storage, and visualization interface.

## 🎯 Project Objectives

- **ETA Prediction** = Customer satisfaction
- **Productivity** = Reduce operational costs  
- **Anomaly Detection** = Avoid hidden losses
- **Problem Areas Identification** = Strategic logistics network planning

## 📊 Dataset

The **Amazon Delivery Dataset** provides a comprehensive view of last-mile logistics operations, including:
- **43,632 deliveries** across multiple cities
- Order details and delivery agents information
- Weather and traffic conditions
- Delivery performance metrics

## 🏗️ Project Architecture

```
├── data/                       # Raw and processed data
│   ├── raw/                    # Original data from Kaggle
│   └── processed/              # Cleaned and transformed data
├── notebooks/                  # Exploratory Data Analysis (EDA)
├── src/                        # Project source code
│   ├── api_server.py           # API for data collection
│   ├── mlflow_server.py        # MLflow server for tracking
│   ├── data/                   # Processing modules
│   ├── models/                 # ML models
│   ├── utils/                  # Utilities
│   └── visualization/          # Visualizations
├── mlruns/                     # MLflow experiments
├── reports/                    # Reports and figures
├── tests/                      # Project tests
└── app.py                      # Streamlit dashboard
```

## 🚀 Implemented Features

### 1. **Data Pipeline**
- ✅ Data collection from Kaggle
- ✅ Data processing and cleaning
- ✅ Storage in organized structure
- ✅ Categorical variable mappings

### 2. **Machine Learning Model**
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Training with Random Forest Regressor
- ✅ Experiment tracking with MLflow
- ✅ Model versioning

### 3. **API and Services**
- ✅ API server for data collection
- ✅ MLflow server for model management
- ✅ Endpoints for predictions

### 4. **User Interface**
- ✅ Interactive dashboard in Streamlit
- ✅ Data and results visualizations
- ✅ Real-time prediction interface

## 🛠️ Technologies Used

- **Python 3.12.11**
- **Anaconda** for environment management
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for ML modeling
- **MLflow** for experiment tracking
- **Streamlit** for interactive dashboard
- **Matplotlib & Seaborn** for visualizations

## ⚙️ Setup and Installation

### Prerequisites
- Anaconda installed
- Python 3.12.11

### Step by step:

1. **Clone the repository:**
```bash
git clone https://github.com/IgorComune/tech_challenge_ml_engineer_phase3.git
cd tech_challenge_ml_engineer_phase3
```

2. **Create a conda environment:**
```bash
conda create -n tech_challenge python=3.12.11
conda activate tech_challenge
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the project:**

**Streamlit Dashboard:**
```bash
streamlit run app.py
```

## 📈 Results and Insights

### Exploratory Analysis
- Identification of seasonal patterns in deliveries
- Correlation between weather conditions and delivery time
- Traffic impact on logistics performance

### Model Performance
- Random Forest model for ETA prediction
- Evaluation metrics available in MLflow
- Feature importance visualization

### Interactive Dashboard
- User-friendly interface for data analysis
- Real-time predictions
- Interactive result visualizations

## 📁 Data Structure

### Processed Data:
- `dataframe_final.csv` - Complete processed dataset
- `dataframe_encoded.csv` - Data with categorical encoding
- `dataframe_encoded_distance.csv` - Data with distance feature
- `mappings/` - Categorical variable mappings

### Models:
- `mlruns/models/rfr_model_v1/` - Versioned Random Forest model

## 🧪 Testing and Validation
- Test notebooks available in `tests/`

## 🤝 Contribution

This project was developed as part of Phase 3 Tech Challenge. Feedback and suggestions are welcome!

## 📄 License

This project is under the license specified in the `LICENSE` file.

---

**Project:** Tech Challenge ML Engineer - Phase 3  
**Institution:** Pós-Tech

---

*"Transforming data into insights, insights into value."*