# Disease-prediction.
A toolkit for disease prediction with preprocessing,models,evaluation,visualizations , and test cases
This toolkit provides data preprocessing, machine learning models, evaluation metrics, and visualization tools for disease prediction.

## Features
- Data preprocessing (cleaning, encoding, normalization)
- Machine learning models (Logistic Regression, Random Forest, etc.)
- Evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Visualizations for insights and model performance
- Test cases for reproducibility
python scripts/preprocessing.py
python scripts/models.py
python scripts/evaluation.py
pandas
numpy
scikit-learn
matplotlib
seaborn
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df
    
