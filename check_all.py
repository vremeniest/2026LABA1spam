#!/usr/bin/env python3
"""
Полная проверка проекта в VS Code
Проверяет: датасет, обучение, тесты, Docker, CI/CD
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_ok(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}📌 {text}{RESET}")

# ============================================================
# 1. ПРОВЕРКА ДАТАСЕТА
# ============================================================
def check_dataset():
    print_header("1. ПРОВЕРКА ДАТАСЕТА SMS SPAM COLLECTION")
    
    # Пути к файлам датасета
    possible_paths = [
        'data/raw/sms.tsv',
        'data/sms.tsv', 
        'spam.csv',
        'data/spam.csv'
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print_error("Датасет не найден! Проверьте пути:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    print_ok(f"Датасет найден: {dataset_path}")
    
    # Читаем датасет
    try:
        if dataset_path.endswith('.tsv'):
            df = pd.read_csv(dataset_path, sep='\t', names=['label', 'message'])
        else:
            df = pd.read_csv(dataset_path)
        
        print_info(f"Всего сообщений: {len(df)}")
        print_info(f"Колонки: {list(df.columns)}")
        
        # Статистика по меткам
        if 'label' in df.columns:
            spam_count = len(df[df['label'] == 'spam'])
            ham_count = len(df[df['label'] == 'ham'])
            print_info(f"Spam: {spam_count} ({spam_count/len(df)*100:.1f}%)")
            print_info(f"Ham: {ham_count} ({ham_count/len(df)*100:.1f}%)")
        
        # Показываем 3 примера
        print("\nПримеры сообщений:")
        for i in range(min(3, len(df))):
            label = df.iloc[i]['label']
            msg = df.iloc[i]['message'][:60] + "..."
            color = GREEN if label == 'ham' else RED
            print(f"  {color}{label}{RESET}: {msg}")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка чтения датасета: {e}")
        return False

# ============================================================
# 2. ПРОВЕРКА МОДЕЛИ (если уже обучена)
# ============================================================
def check_model():
    print_header("2. ПРОВЕРКА МОДЕЛИ MODERNBERT")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Проверяем, есть ли сохраненная модель
        model_paths = [
            './coco_location_model',
            './saved_model',
            './model'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            print_ok(f"Найдена сохраненная модель: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print_info("Сохраненная модель не найдена. Загружаем ModernBERT...")
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
            model = AutoModelForSequenceClassification.from_pretrained(
                "answerdotai/ModernBERT-base", 
                num_labels=2
            )
        
        print_ok(f"Модель загружена. Параметров: {model.num_parameters():,}")
        
        # Тестируем на примерах
        test_messages = [
            ("FREE iPhone! Click here to claim", "spam"),
            ("I'll be there in 10 minutes", "ham"),
            ("CONGRATULATIONS! You won $1000", "spam"),
            ("See you tomorrow at the cafe", "ham"),
        ]
        
        print("\nТестовые предсказания:")
        model.eval()
        for msg, expected in test_messages:
            inputs = tokenizer(msg, return_tensors='pt', truncation=True, max_length=64)
            with torch.no_grad():
                pred = model(**inputs).logits.argmax().item()
                pred_label = "spam" if pred == 1 else "ham"
            
            color = GREEN if pred_label == expected else RED
            print(f"  {color}Текст: {msg[:40]}...{RESET}")
            print(f"    Ожидается: {expected} → Предсказание: {pred_label}")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка при проверке модели: {e}")
        return False

# ============================================================
# 3. ПРОВЕРКА DOCKER
# ============================================================
def check_docker():
    print_header("3. ПРОВЕРКА DOCKER КОНТЕЙНЕРИЗАЦИИ")
    
    # Проверяем наличие Dockerfile
    if os.path.exists('Dockerfile'):
        print_ok("Dockerfile найден")
        
        # Показываем содержимое Dockerfile
        with open('Dockerfile', 'r') as f:
            content = f.read()
            if 'FROM python' in content:
                print_ok("  ✅ Базовый образ Python")
            if 'COPY src/' in content:
                print_ok("  ✅ Копирование исходного кода")
            if 'pip install' in content:
                print_ok("  ✅ Установка зависимостей")
            if 'EXPOSE' in content:
                print_ok("  ✅ Открыт порт")
    else:
        print_error("Dockerfile не найден!")
        return False
    
    # Проверяем наличие requirements.txt
    if os.path.exists('requirements.txt'):
        print_ok("requirements.txt найден")
        with open('requirements.txt', 'r') as f:
            reqs = f.read()
            required_packages = ['transformers', 'torch', 'pandas', 'scikit-learn']
            for pkg in required_packages:
                if pkg in reqs:
                    print_ok(f"  ✅ {pkg} в зависимостях")
    else:
        print_error("requirements.txt не найден!")
    
    # Проверяем Docker (если установлен)
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_ok(f"Docker установлен: {result.stdout.strip()}")
            
            # Пробуем собрать образ (опционально)
            print_info("Попытка сборки Docker образа...")
            result = subprocess.run(['docker', 'build', '-t', 'test-model', '.'], 
                                   capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print_ok("Docker образ успешно собран!")
            else:
                print_info("Сборка образа не выполнена (возможно, нужны файлы)")
        else:
            print_info("Docker не установлен в системе")
    except FileNotFoundError:
        print_info("Docker не найден в PATH")
    except subprocess.TimeoutExpired:
        print_info("Сборка образа заняла слишком много времени")

# ============================================================
# 4. ПРОВЕРКА CI/CD
# ============================================================
def check_cicd():
    print_header("4. ПРОВЕРКА CI/CD КОНФИГУРАЦИИ")
    
    # Проверяем GitHub Actions
    cicd_path = '.github/workflows/cicd.yml'
    if os.path.exists(cicd_path):
        print_ok(f"CI/CD конфиг найден: {cicd_path}")
        
        with open(cicd_path, 'r') as f:
            content = f.read()
            
            checks = [
                ('name: CI/CD Pipeline', 'Имя пайплайна'),
                ('on:', 'Триггеры'),
                ('push:', 'Push триггер'),
                ('jobs:', 'Джобы'),
                ('test:', 'Тест джоба'),
                ('docker:', 'Docker джоба'),
                ('docker/build-push-action', 'Docker build action')
            ]
            
            for pattern, desc in checks:
                if pattern in content:
                    print_ok(f"  ✅ {desc}")
                else:
                    print_info(f"⚠️ Не найден: {desc}")
    else:
        print_error(f"CI/CD конфиг не найден: {cicd_path}")
        return False
    
    return True

# ============================================================
# 5. ПРОВЕРКА ТЕСТОВ
# ============================================================
def check_tests():
    print_header("5. ПРОВЕРКА ТЕСТОВ")
    
    test_dir = 'tests'
    if os.path.exists(test_dir):
        print_ok(f"Папка тестов найдена: {test_dir}")
        
        test_files = list(Path(test_dir).glob('test_*.py'))
        if test_files:
            print_ok(f"Найдено {len(test_files)} тестовых файлов:")
            for tf in test_files:
                print(f"  - {tf.name}")
            
            # Пытаемся запустить тесты
            try:
                result = subprocess.run(['pytest', test_dir, '-v', '--collect-only'], 
                                       capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print_ok("Тесты могут быть запущены")
                else:
                    print_info("Pytest не настроен или нет тестов")
            except FileNotFoundError:
                print_info("pytest не установлен")
        else:
            print_error("Нет тестовых файлов!")
    else:
        print_info("Папка tests/ не найдена")

# ============================================================
# 6. ЗАПУСК ТРЕНИРОВКИ (опционально)
# ============================================================
def run_training():
    print_header("6. ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")
    
    response = input("Запустить обучение модели? (y/N): ")
    if response.lower() != 'y':
        print_info("Обучение пропущено")
        return
    
    if not os.path.exists('src/train.py'):
        print_error("src/train.py не найден!")
        return
    
    try:
        print_info("Запуск обучения...")
        result = subprocess.run(['python', 'src/train.py'], timeout=300)
        if result.returncode == 0:
            print_ok("Обучение успешно завершено!")
        else:
            print_error("Ошибка при обучении")
    except subprocess.TimeoutExpired:
        print_error("Обучение превысило лимит времени (5 минут)")

# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def main():
    print_header="🔍 ПРОВЕРКА ПРОЕКТА VS CODE"
    print(print_header)
    print(f"Директория проекта: {os.getcwd()}")
    
    results = []
    
    # Запускаем все проверки
    results.append(("Датасет", check_dataset()))
    results.append(("Модель", check_model()))
    results.append(("Docker", check_docker()))
    results.append(("CI/CD", check_cicd()))
    results.append(("Тесты", check_tests()))
    
    # Итог
    print_header="📊 ИТОГОВЫЙ РЕЗУЛЬТАТ"
    print(print_header)
    
    passed = sum(1 for _, status in results if status)
    total = len(results)
    
    for name, status in results:
        if status:
            print_ok(f"{name}: УСПЕШНО")
        else:
            print_error(f"{name}: ПРОБЛЕМА")
    
    print(f"\n{'✅' if passed == total else '⚠️'} ВСЕГО: {passed}/{total} пунктов выполнено")
    
    if passed == total:
        print(f"\n{GREEN}🎉 ПОЗДРАВЛЯЮ! Все проверки пройдены успешно! 🎉{RESET}")
    else:
        print(f"\n{YELLOW}📝 Исправьте отмеченные проблемы и запустите скрипт снова{RESET}")
    
    # Предложение запустить обучение
    if passed >= 3:
        run_training()

if __name__ == "__main__":
    main()