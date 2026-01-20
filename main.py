from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# load model
model = joblib.load("model_kelulusan.pkl")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def predict(request: Request):
    form = await request.form()

    math = float(form["math"])
    reading = float(form["reading"])
    writing = float(form["writing"])

    input_data = pd.DataFrame([{
        "math score": math,
        "reading score": reading,
        "writing score": writing
    }])

    prediction = model.predict(input_data)[0]
    result = "Lulus" if prediction == 1 else "Tidak Lulus"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
