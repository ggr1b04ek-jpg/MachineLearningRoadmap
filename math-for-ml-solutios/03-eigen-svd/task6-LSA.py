import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

with open(r'C:\Users\user\MachineLearningRoadmap\math-for-ml-solutios\03-eigen-svd\docs.txt', 'r', encoding='utf-8') as f:
    documents = [line.strip() for line in f if line.strip()]

if len(documents) < 2:
    raise ValueError("Для построения TF-IDF нужно минимум 2 документа.")

RU_STOP_WORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 
    'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 
    'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 
    'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 
    'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 
    'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 
    'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 
    'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 
    'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 
    'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 
    'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 
    'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 
    'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 
    'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
]

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=RU_STOP_WORDS
)

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print(f"Размер матрицы: {tfidf_matrix.shape}")
print(f"Размер словаря: {len(feature_names)} терминов")
print(f"Плотность: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) * 100:.2f}%")

k = 10

U, sigma, Vt = np.linalg.svd(tfidf_matrix.toarray().astype(np.float64), full_matrices=False)
U_k = U[:, :k]
sigma_k = sigma[:k]
Vt_k = Vt[:k, :]

lsa_matrix = U_k @ np.diag(sigma_k)

row_norms = np.linalg.norm(lsa_matrix, axis=1, keepdims=True)
lsa_matrix_norm = lsa_matrix / (row_norms + 1e-10)

explained_var = np.sum(sigma_k**2) / np.sum(sigma**2)
print(f"Исходная размерность: {tfidf_matrix.shape[1]} терминов")
print(f"Размерность LSA: {k} латентных тем")
print(f"Объяснённая дисперсия: {explained_var:.2%}")

similarity_matrix = lsa_matrix_norm @ lsa_matrix_norm.T

print("\nДля каждого документа топ-3 наиболее похожих:")
print("-" * 80)

for i in range(len(documents)):
    similar_indices = np.argsort(similarity_matrix[i])[::-1][1:4]
    
    doc_preview = documents[i][:70] + "..." if len(documents[i]) > 70 else documents[i]
    print(f"\n[{i+1:03d}] \"{doc_preview}\"")
    
    for rank, j in enumerate(similar_indices, 1):
        score = similarity_matrix[i][j]
        sim_doc = documents[j][:70] + "..." if len(documents[j]) > 70 else documents[j]
        print(f"   {rank}. Сходство {score:.3f} → [{j+1:03d}] \"{sim_doc}\"")