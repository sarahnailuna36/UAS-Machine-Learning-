import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ======================
# LOAD DATASET
# ======================
data = pd.read_excel("data.xlsx")

# ======================
# PREPROCESSING
# ======================

# Buat target kelulusan
data["average_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

data["status"] = data["average_score"].apply(
    lambda x: 1 if x >= 70 else 0
)

# Gunakan HANYA 3 fitur (sesuai web)
X = data[["math score", "reading score", "writing score"]]
y = data["status"]

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ======================
# TRAIN MODEL
# ======================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# EVALUASI
# ======================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# ======================
# SIMPAN MODEL
# ======================
joblib.dump(model, "model_kelulusan.pkl")
print("Model saved as model_kelulusan.pkl")
