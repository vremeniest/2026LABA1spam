import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# --- Lifespan для загрузки/выгрузки модели ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Выполняется ПЕРЕД запуском приложения (startup)
    global model, vectorizer
    print("🚀 Загрузка модели...")
    with open('models/spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Модель и векторизатор загружены")
    yield
    # Выполняется ПОСЛЕ остановки приложения (shutdown)
    print("🛑 Выгрузка модели...")
    model = None
    vectorizer = None

# --- Инициализация приложения ---
app = FastAPI(
    title="SMS Spam Detector API",
    description="API для определения спама в SMS сообщениях",
    version="1.0.0",
    lifespan=lifespan  # <-- Современный способ
)

# Глобальные переменные для модели
model = None
vectorizer = None

# --- Модели данных ---
class SMSRequest(BaseModel):
    text: str

class SMSResponse(BaseModel):
    is_spam: bool
    label: str
    confidence: float

# --- Эндпоинты ---
@app.get("/")
def root():
    return {"message": "SMS Spam Detector API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=SMSResponse)
def predict(request: SMSRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    X = vectorizer.transform([request.text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(probabilities[prediction])

    return SMSResponse(
        is_spam=bool(prediction),
        label="spam" if prediction == 1 else "ham",
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)