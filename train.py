import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_excel("data.xlsx")

# Buat target kelulusan
data["average_score"] = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

data["status"] = data["average_score"].apply(
    lambda x: 1 if x >= 70 else 0
)

# Fitur & target
X = data[["math score", "reading score", "writing score"]]
y = data["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Simpan model
joblib.dump(model, "model_kelulusan.pkl")
print("Model berhasil disimpan")
