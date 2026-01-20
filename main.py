from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load("model_kelulusan.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/", response_class=HTMLResponse)
def predict(
    request: Request,
    math: float = Form(...),
    reading: float = Form(...),
    writing: float = Form(...)
):
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
