from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from pyrsistent import PClass

from .ml_model import predict_survival_chance


class ModelData(BaseModel):
    Pclass: int
    Sex: int
    Age: int
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

app = FastAPI()

@app.post('/predict')
def predict_survival_chance_api(data: ModelData):
    input_data = data.dict()
    prediction = predict_survival_chance(input_data)
    return {'survival_chance': prediction}

