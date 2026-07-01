from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
from typing import List
import uvicorn 
import os

# --- 1. Veri Modelleri (C#\'tan Gelecek JSON Formatı) ---
class DailyData(BaseModel):
    sales: float = Field(..., description="Günlük toplam satış adedi")
    avg_temp: float = Field(..., description="Ortalama sıcaklık")
    avg_humidity: float = Field(..., description="Ortalama nem")
    normalizedDensity: float = Field(..., description="Bölgenin nüfus yoğunluğu (0-1 arası)")

class PredictionRequest(BaseModel):
    city: str
    town: str
    features: List[DailyData] = Field(..., min_length=7, max_length=7, description="Son 7 günün verisi olmak zorundadır")

# --- 2. FastAPI Uygulaması ve Model Yükleme ---
app = FastAPI(title="ESEN Outbreak Prediction API", version="1.0")

# Modeli global olarak bir kere yüklüyoruz (Her istekte tekrar yüklenmesin diye)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "esen_regional_lstm_model.keras"),
    os.path.join(os.getcwd(), "esen_regional_lstm_model.keras"),
]


def load_model_from_candidates(candidate_paths):
    last_error = None

    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue

        try:
            loaded_model = tf.keras.models.load_model(candidate_path, compile=False)
            print(f"LSTM model loaded successfully: {candidate_path}")
            return loaded_model
        except Exception as error:
            last_error = error
            print(f"Model load failed for {candidate_path}: {error}")

    print("Model file not found in any expected location.")
    if last_error is not None:
        print(f"Last load error: {last_error}")
    return None


model = load_model_from_candidates(MODEL_CANDIDATES)

THRESHOLD = 0.354  # Belirlediğimiz kritik salgın eşiği

# --- 3. Tahmin Uç Noktası (Endpoint) ---
@app.post("/api/predict")
async def predict_outbreak(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Yapay zeka modeli aktif değil.")

    try:
        # C#\'tan gelen JSON verisini Numpy dizisine çeviriyoruz
        # Verilerin zaten normalize edilmiş olduğu varsayılır.
        input_data = np.array([[day.sales, day.avg_temp, day.avg_humidity, day.normalizedDensity] for day in request.features])

        # Modeli [1, 7, 4] formatına (1 batch, 7 gün, 4 özellik) sokuyoruz
        X_input = np.expand_dims(input_data, axis=0)
        
        # Tahmin yap
        # Model.predict doğrudan numpy dizisi döndürür, bu yüzden .tolist() kullanıyoruz.
        prediction_value = float(model.predict(X_input, verbose=0)[0].tolist()[0])
        is_outbreak = prediction_value > THRESHOLD

        # C# Backend\'ine gidecek olan final raporu
        return {
            "region": f"{request.city} - {request.town}",
            "risk_score": round(prediction_value, 4),
            "threshold": THRESHOLD,
            "is_outbreak_detected": is_outbreak,
            "message": "Salgın riski tespit edildi! Acil aksiyon alınmalı." if is_outbreak else "Bölgede anomali yok."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin işlemi sırasında hata: {str(e)}")

# Servisi çalıştırmak için
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
