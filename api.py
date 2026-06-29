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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "esen_regional_lstm_model.keras")

# Fallback for Render deployment where model might be in the root directory
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "esen_regional_lstm_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ LSTM Modeli başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model yüklenirken hata oluştu: {e}. Denenen yol: {MODEL_PATH}")
    model = None

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
