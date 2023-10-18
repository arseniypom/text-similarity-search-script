import os
import json
import numpy as np
import pymorphy2
import nltk
import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from synonyms import synonyms

# Загрузка стоп-слов
nltk.download("stopwords")

# Установка переменной окружения
os.environ["TORCH_HOME"] = "./caches/cache7"

# Загрузка данных
with open("./general.json", "r", encoding="utf-8") as file:
    paragraphs = json.load(file)

# Инициализация лемматизатора
morph = pymorphy2.MorphAnalyzer()

# Функция для лемматизации текста
def lemmatize(text):
    words = nltk.word_tokenize(text)
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)


# Функция для расширения запроса синонимами
def expand_query_with_synonyms(query, synonyms):
    lemmatized_words = lemmatize(query).split()
    expanded_query = []
    for word in lemmatized_words:
        expanded_query.append(word)
        if word in synonyms:
            expanded_query.extend(synonyms[word])
    return " ".join(expanded_query)

# Задание запроса и расширение его синонимами
query = "Какую макулатуру вы принимаете?"
expanded_query = expand_query_with_synonyms(query, synonyms)

# Токенизация и лемматизация текста
lemmatized_paragraphs = [lemmatize(paragraph) for paragraph in paragraphs]
lemmatized_query = lemmatize(expanded_query)

# Векторизация текста с использованием TfidfVectorizer
stop_words = set(stopwords.words('russian'))
lemmatized_stop_words = [morph.parse(word)[0].normal_form for word in stop_words]
vectorizer = TfidfVectorizer(stop_words=list(lemmatized_stop_words))

X = vectorizer.fit_transform(lemmatized_paragraphs)
query_tfidf_vector = vectorizer.transform([lemmatized_query])

# Compute TF-IDF similarities
tfidf_similarities = np.dot(X, query_tfidf_vector.T).toarray().flatten()

# Инициализация модели Universal Sentence Encoder
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Функция для вычисления эмбеддингов с использованием USE
def compute_embeddings(texts):
    return model(texts).numpy()

# Вычисление эмбеддингов для абзацев и запроса
paragraph_embeddings = compute_embeddings(lemmatized_paragraphs)
query_embedding = compute_embeddings([lemmatized_query])[0]

# Вычисление и комбинирование схожестей
neural_similarities = [
    1 - cosine(query_embedding, paragraph_embedding)
    for paragraph_embedding in paragraph_embeddings
]

combined_similarities = [
    0.3 * tfidf_sim + 0.7 * neural_sim
    for tfidf_sim, neural_sim in zip(tfidf_similarities, neural_similarities)
]

# Получение лучшего совпадения
indexed_similarities = list(enumerate(combined_similarities))
sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)


# Список запросов для поиска
queries = [
    "Какую макулатуру вы принимаете?",
    "Какие требования к сдаче бумаги?",
    "Могу ли я сдать втулки?",
    "Какие виды отходов принимаются?",
    "Какие предметы из алюминия можно сдать?",
    "Какие требования к сдаче картонных изделий?",
    "Какие ограничения на ламинированные изделия?"
]

# Функция для поиска и записи результатов в файл
def search_and_write_to_file(query, file):
    expanded_query = expand_query_with_synonyms(query, synonyms)
    lemmatized_query = lemmatize(expanded_query)
    query_tfidf_vector = vectorizer.transform([lemmatized_query])
    tfidf_similarities = np.dot(X, query_tfidf_vector.T).toarray().flatten()
    
    query_embedding = model.encode(lemmatized_query)
    neural_similarities = [
        1 - cosine(query_embedding, paragraph_embedding)
        for paragraph_embedding in paragraph_embeddings
    ]
    
    combined_similarities = [
        0.3 * tfidf_sim + 0.7 * neural_sim
        for tfidf_sim, neural_sim in zip(tfidf_similarities, neural_similarities)
    ]
    
    indexed_similarities = list(enumerate(combined_similarities))
    sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
    
    file.write(f"QUERY: {query}\n")
    if sorted_similarities:
        for i, (index, similarity) in enumerate(sorted_similarities[:3]):
            file.write(f"------- {i + 1} BEST: {paragraphs[index]}\n")
    file.write("===\n")

# Открытие файла и выполнение поиска для каждого запроса
with open("search_results7.txt", "w", encoding="utf-8") as file:
    for query in queries:
        search_and_write_to_file(query, file)