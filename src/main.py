from data import load_data, split_data
from model import build_model
from evaluate import evaluate

def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, X_test, y_test, feature_names)
    print

main()

import json

with open('outputs/metrics.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


print("The model is based on Logistic Regression","&","Accuracy over the model :", round(list(data.values())[0]*100, 2),"%")





