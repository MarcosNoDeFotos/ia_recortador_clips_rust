import os
import joblib
import numpy as np
from sklearn.metrics import classification_report

CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"


data = joblib.load(CURRENT_PATH+"modelo_rust_svm.pkl")
clf = data["clf"]
classes = data["classes"]
X_test = data["X_test"]
y_test = data["y_test"]

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes))
