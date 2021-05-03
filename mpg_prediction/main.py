from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from pydantic.main import BaseModel
import joblib
from typing import List
from mpg_prediction.process import dict_to_df,pipeline_transformer

# class Settings(BaseSettings):
#     model_dir: str

app = FastAPI()
# settings = Settings()

# Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Load model
# config = {
#     "model_output_path": settings.model_dir,
#     "featurizer_output_path": settings.model_dir
# }


##vehicle config
# vehicle_config = {
#     'Cylinders': [4, 6, 8],
#     'Displacement': [155.0, 160.0, 165.5],
#     'Horsepower': [93.0, 130.0, 98.0],
#     'Weight': [2500.0, 3150.0, 2600.0],
#     'Acceleration': [15.0, 14.0, 16.0],
#     'Model_Year': [81, 80, 78],
#     'Origin': [3, 2, 1]
# }
# requests.post('http://127.0.0.1:8000/api/predict',json=vehicle_config).content


class mpg_columns(BaseModel):
    Cylinders: List[int]=list() 
    Displacement:List[float]=list() 
    Horsepower:List[float]=list()
    Weight: List[float]=list() 
    Acceleration:List[float]=list()
    Model_Year: List[int]=list()
    Origin:List[int]=list() 

class PredictResponse(BaseModel):
    data: List[float]


@app.post("/api/predict",response_model=PredictResponse)
async def predict_mpg(mpg: mpg_columns):
    data = mpg.dict()
    process= dict_to_df(data)
    prep_data = pipeline_transformer(process)
    model = joblib.load('./mpg_prediction/model/rand_model.pkl')    
    predictions = model.predict(prep_data)
    pred_list=list(predictions)
    # print(pred_list)
    return PredictResponse(data=pred_list)
    