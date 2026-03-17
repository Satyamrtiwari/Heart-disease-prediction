# Heart Disease Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Django](https://img.shields.io/badge/Django-REST-green)
![React](https://img.shields.io/badge/React-Frontend-blue)
![AI](https://img.shields.io/badge/AI-HuggingFace-orange)

A machine learning-powered clinical decision support tool for cardiovascular risk assessment using patient clinical parameters.

## 📌 Overview

This web application provides healthcare professionals and researchers with an automated risk stratification system for heart disease prediction. The system analyzes patient demographics, vital signs, laboratory results, and clinical symptoms to generate evidence-based risk assessments.

## 🎯 Features

- **Interactive Risk Assessment**: Streamlined interface for inputting patient clinical data
- **Real-time Predictions**: Instant risk classification with confidence scoring
- **Clinical Guidelines**: Integrated reference values and medical standards
- **Professional Interface**: Clean, medical-grade user experience
- **Confidence Metrics**: Visual gauge showing prediction reliability

## 🛠️ Technology Stack

- **Frontend**: Streamlit web framework
- **Machine Learning**: K-Nearest Neighbors (KNN) algorithm
- **Data Processing**: Pandas, Scikit-learn
- **Visualization**: Plotly for interactive charts
- **Deployment**: Python-based web application

## Clinical Parameters

The system evaluates the following clinical inputs:

### Demographics
- Age and gender

### Vital Signs & Laboratory
- Resting blood pressure
- Serum cholesterol levels
- Fasting blood glucose
- Maximum heart rate achieved

### Diagnostic Tests
- Resting electrocardiogram (ECG) findings
- Exercise-induced ST depression (Oldpeak)
- ST slope characteristics

### Clinical Symptoms
- Chest pain classification
- Exercise-induced angina presence

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-assessment.git
cd heart-disease-assessment
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are present:
- `knn_heart_model.pkl`
- `heart_scaler.pkl`
- `heart_columns.pkl`

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Navigate to the **Risk Assessment** tab
2. Input patient clinical parameters in the respective sections
3. Click **Generate Risk Assessment** to process the evaluation
4. Review the risk classification and recommended actions
5. Reference the **Guidelines** tab for clinical standards

## Model Performance

The KNN model has been trained on validated cardiovascular datasets with clinical parameter standardization and feature scaling for optimal prediction accuracy.

## Important Disclaimers

- **Educational Use Only**: This tool is designed for educational and screening purposes
- **Not for Diagnosis**: Results should not replace professional medical consultation
- **Clinical Judgment**: Always consult qualified healthcare providers for medical decisions
- **Research Tool**: Intended to support clinical decision-making, not replace it

## 📂 File Structure

```
├── app.py                    # Main Streamlit application
├── knn_heart_model.pkl      # Trained KNN model
├── heart_scaler.pkl         # Feature scaling parameters
├── heart_columns.pkl        # Expected feature columns
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
plotly>=5.15.0
joblib>=1.3.0
```

## Contributing

Contributions are welcome for improving the clinical accuracy, user interface, or adding new features. Please ensure all modifications maintain medical standards and professional presentation.

## License

This project is intended for educational and research purposes. Please consult with medical professionals before any clinical application.

## 👤 Author

**Developer**: **Satyam Tiwari**
**Purpose**: Clinical Decision Support System  
**Status**: Educational/Research Tool

---

*This system is designed to assist healthcare professionals and should not be used as the sole basis for medical decisions.*
