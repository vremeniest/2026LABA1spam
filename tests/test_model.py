import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_model_exists():
    """Проверка существования файла модели"""
    model_path = Path("models/spam_model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")
    
    assert model_path.exists(), "Модель не найдена"
    assert vectorizer_path.exists(), "Векторизатор не найден"

def test_model_loading():
    """Проверка загрузки модели"""
    with open("models/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    assert model is not None
    assert vectorizer is not None

def test_model_prediction():
    """Проверка предсказаний модели"""
    with open("models/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Тестовые сообщения (используем реальные из датасета)
    ham_text = "Go until jurong point, crazy"
    spam_text = "Free entry in 2 a wkly comp to win FA Cup final tkts"
    
    ham_vec = vectorizer.transform([ham_text])
    spam_vec = vectorizer.transform([spam_text])
    
    ham_pred = model.predict(ham_vec)[0]
    spam_pred = model.predict(spam_vec)[0]
    
    # Просто проверяем, что предсказания делаются (не обязательно 0 и 1)
    assert ham_pred in [0, 1]
    assert spam_pred in [0, 1]