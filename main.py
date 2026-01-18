from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model_kelulusan.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        math = float(request.form["math"])
        reading = float(request.form["reading"])
        writing = float(request.form["writing"])

        input_data = pd.DataFrame([{
            "math score": math,
            "reading score": reading,
            "writing score": writing
        }])

        prediction = model.predict(input_data)[0]
        result = "Lulus" if prediction == 1 else "Tidak Lulus"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
