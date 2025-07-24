from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_event_type

app = FastAPI(title="NOAA Weather Event Detector 🚀")

class EventInput(BaseModel):
    lat: float
    lon: float
    state_code: int
    month: int

@app.get("/")
def root():
    return {"message": "🌩️ NOAA Weather Event Detection API is running!"}

@app.post("/predict")
def predict_event(input_data: EventInput):
    try:
        prediction = predict_event_type(
            lat=input_data.lat,
            lon=input_data.lon,
            state_code=input_data.state_code,
            month=input_data.month
        )
        return {
            "predicted_event_type": prediction,
            "input": input_data.dict()
        }
    except Exception as e:
        return {"error": str(e)}
