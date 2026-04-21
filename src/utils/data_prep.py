import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess(data_path="data/raw/sms.tsv", test_size=0.2, random_state=42):
    """
    Загрузка и предобработка SMS датасета
    
    Returns:
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    """
    logger.info(f"Загрузка данных из {data_path}")
    
    # Загрузка данных
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    # Преобразование меток: spam -> 1, ham -> 0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    logger.info(f"Всего сообщений: {len(df)}")
    logger.info(f"Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']
    )
    
    # Векторизация текста
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    logger.info(f"Train размер: {X_train_tfidf.shape}")
    logger.info(f"Test размер: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer