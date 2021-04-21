from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from pydantic.main import BaseModel
import joblib
import uvicorn
from typing import Optional,Set,List


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
# model = RandomForestModel(config)

##vehicle config
# vehicle_config = {
#     'Cylinders': [4, 6, 8],
#     'Displacement': [155.0, 160.0, 165.5],
#     'Horsepower': [93.0, 130.0, 98.0],
#     'Weight': [2500.0, 3150.0, 2600.0],
#     'Acceleration': [15.0, 14.0, 16.0],
#     'Model Year': [81, 80, 78],
#     'Origin': [3, 2, 1]
# }

# class mpg_columns(BaseModel):
#     Cylinders: int 
#     Displacement:float 
#     Horsepower:float
#     Weight: float 
#     Acceleration:float
#     Model_Year: int
#     Origin:int 

class mpg_columns(BaseModel):
    Cylinders: List[int]=list() 
    Displacement:List[float]=list() 
    Horsepower:List[float]=list()
    Weight: List[float]=list() 
    Acceleration:List[float]=list()
    Model_Year: List[int]=list()
    Origin:List[int]=list() 


# class Prediction(BaseModel):
#     label: float
#     probs: List[float]
from process import dict_to_df,pipeline_transformer

@app.post("/api/predict")
async def predict_mpg(mpg: mpg_columns, request:Request):
    data = mpg.dict()
    process= dict_to_df(data)
    prep_data = pipeline_transformer(process)
    model = joblib.load('./model_checkpoints/rand_model.pkl')    
    predictions = model.predict(prep_data)
    print(list(predictions))
    return 
    

# @app.post("/api/predict")
# def predict_mpg(mpg: mpg_columns):
#     data = mpg.dict()
#     print('data',data)
#     process= dict_to_df(data)
#     prep_data = pipeline_transformer(process)
#     # print(prep_data)
#     print(process.head())
#     model = joblib.load('./model_checkpoints/rand_model.pkl')
    
#     probability = model.predict(prep_data)
#     return probability
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)