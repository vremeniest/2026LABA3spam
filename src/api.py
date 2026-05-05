import pickle
import os
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# ========== ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ ==========
def get_db_connection():
    """Создаёт подключение к PostgreSQL из переменных окружения"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'spam_db'),
        user=os.getenv('DB_USER', 'spam_user'),
        password=os.getenv('DB_PASSWORD', 'spam_password')
    )

def init_db():
    """Создаёт таблицу для хранения предсказаний, если её нет"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                text TEXT,
                label TEXT,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Таблица predictions готова")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации БД: {e}")

# ========== ЗАГРУЗКА МОДЕЛИ (LIFESPAN) ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Выполняется ПЕРЕД запуском приложения (startup)
    global model, vectorizer
    print("🚀 Загрузка модели...")
    
    # Проверяем, существуют ли файлы модели
    model_path = Path('models/spam_model.pkl')
    vectorizer_path = Path('models/vectorizer.pkl')
    
    if not model_path.exists() or not vectorizer_path.exists():
        print("⚠️ Файлы модели не найдены! Сначала запустите train.py")
        model = None
        vectorizer = None
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Модель и векторизатор загружены")
    
    # Инициализируем базу данных
    init_db()
    
    yield
    
    # Выполняется ПОСЛЕ остановки приложения (shutdown)
    print("🛑 Выгрузка модели...")
    model = None
    vectorizer = None

# ========== ИНИЦИАЛИЗАЦИЯ FASTAPI ==========
app = FastAPI(
    title="SMS Spam Detector API",
    description="API для определения спама в SMS сообщениях. Сохраняет предсказания в PostgreSQL.",
    version="2.0.0",
    lifespan=lifespan
)

# Глобальные переменные для модели
model = None
vectorizer = None

# ========== МОДЕЛИ ДАННЫХ ДЛЯ API ==========
class SMSRequest(BaseModel):
    text: str

class SMSResponse(BaseModel):
    is_spam: bool
    label: str
    confidence: float

# ========== ЭНДПОИНТЫ ==========
@app.get("/")
def root():
    return {
        "message": "SMS Spam Detector API",
        "status": "running",
        "database": "PostgreSQL"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=SMSResponse)
def predict(request: SMSRequest):
    # Проверка, что модель загружена
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Запустите train.py для обучения.")
    
    # Предсказание
    X = vectorizer.transform([request.text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(probabilities[prediction])
    label = "spam" if prediction == 1 else "ham"
    
    # Сохранение предсказания в базу данных
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (text, label, confidence) VALUES (%s, %s, %s)",
            (request.text, label, confidence)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Предсказание сохранено в БД: {label} (confidence: {confidence:.3f})")
    except Exception as e:
        # Логируем ошибку, но не прерываем работу API
        print(f"⚠️ Ошибка сохранения в БД: {e}")
    
    return SMSResponse(
        is_spam=bool(prediction),
        label=label,
        confidence=confidence
    )

# ========== ЗАПУСК ПРИЛОЖЕНИЯ (ДЛЯ ЛОКАЛЬНОГО ТЕСТИРОВАНИЯ) ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)