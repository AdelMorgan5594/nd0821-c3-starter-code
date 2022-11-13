# Put the code for your API here.
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics,inference




app = FastAPI()

@app.get("/")
async def root():
    return {"message": "welcome to the API"}


class dataInput(BaseModel):
    age: int = Field(..., example=35)
    workclass: str = Field(..., example="Federal-gov")
    fnlgt: int = Field(..., example=249409)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., alias="education-num", example=9)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Other-service")
    relationship: str = Field(..., example="Own-child")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )
    

@app.post("/inference/")
async def predict(item: dataInput):
    cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    X = pd.DataFrame(data = [item.dict(by_alias=True)], index = [0])
    X,_,_,_ = process_data(X, cat_features, label=None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)

    if pred[0]:
        pred = 'salary': '>50k'
    else:
        pred = 'salary': '<=50k'
    return pred


