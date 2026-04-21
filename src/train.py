import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь Python
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import configparser
from pathlib import Path
from src.utils.data_prep import load_and_preprocess

def main():
    # Загрузка конфигурации
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Параметры модели
    C = float(config['MODEL'].get('C', '1.0'))
    max_iter = int(config['MODEL'].get('MAX_ITER', '100'))
    
    print("📊 Загрузка и подготовка данных...")
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()
    
    print("🤖 Обучение модели Logistic Regression...")
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка качества
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Точность модели: {accuracy:.4f}")
    print("\n📋 Отчет классификации:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Сохранение модели и векторизатора
    Path('models').mkdir(exist_ok=True)
    
    with open('models/spam_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n💾 Модель сохранена в models/spam_model.pkl")
    print("💾 Векторизатор сохранён в models/vectorizer.pkl")

if __name__ == "__main__":
    main()