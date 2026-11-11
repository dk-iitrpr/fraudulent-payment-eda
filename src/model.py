# src/model.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
