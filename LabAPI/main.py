import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI
import json
from pyod.models.knn import KNN

app = FastAPI()

knn = None
clf = None


@app.on_event("startup")
def load_train_model():
    global knn
    global clf
    df = pd.read_csv("iris_processed.csv")
    knn = KNeighborsClassifier(n_neighbors=len(np.unique(df["y"])))
    clf = KNN()
    clf.fit(df[df.columns[:4]].values.tolist(), df["y"])
    knn.fit(df[df.columns[:4]].values.tolist(), df["y"])
    print("Model finished the training")


@app.post("/predict")
def predict(p1: float, p2: float, p3: float, p4: float):
    pred = knn.predict([[p1, p2, p3, p4]])
    response = {"prediction": pred.tolist()[0]}
    return json.dumps(response)


@app.post("/predict_anomaly")
def predict_anomaly(p1: float, p2: float, p3: float, p4: float):
    pred = clf.predict([[p1, p2, p3, p4]])
    pred_score = clf.decision_function([[p1, p2, p3, p4]])
    response = {
        "is-anomaly": pred.tolist()[0],
        "prediction_score": pred_score.tolist()[0],
    }
    return json.dumps(response)


@app.get("/")
def read_root():
    return {"Hello": "World"}
