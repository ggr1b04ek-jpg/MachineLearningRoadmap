def cosine_similarity(v1, v2):
    norm_v1 = sum(x  ** 2 for x in v1) ** 0.5
    norm_v2 = sum(x ** 2 for x in v2) ** 0.5
    dot_product = sum(x * y for x, y in zip(v1, v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

texts = [
    "машинное обучение и нейронные сети это будущее искусственного интеллекта",
    "кошки и собаки самые популярные домашние животные в мире",
    "нейронные сети используются для обработки естественного языка и данных",
    "в лесу растут высокие деревья и поют птицы весной",
    "искусственного интеллект меняет подход к анализу больших данных"
]

all_words = set()
for text in texts:
    for word in text.split():
        all_words.add(word)

vocabulary = sorted(list(all_words))
print(f"Всего слов: {len(vocabulary)}")

def text_to_vector(text, vocab):
    words = text.split()
    vector = [0] * len(vocab)
    for i, word in enumerate(vocab):
        vector[i] = words.count(word)
    return vector
    
vectors = [text_to_vector(t, vocabulary) for t in texts]

results = []
n = len(texts)

for i in range(n):
    for j in range(i+1, n):
        sim = cosine_similarity(vectors[i], vectors[j])
        results.append((i + 1, j + 1, sim))
    
        print(f"Текст {i+1} <-> Текст {j+1}: {sim:.4f}")

results.sort(key=lambda x: x[2], reverse=True)
for i, j, score in results:
    print(f"{i}. Пара текстов [{i}, {j}] -> Близость: {score:.4f}")
    print(f"    Текст {i}: \"{texts[i-1][:50]}...\"")
    print(f"    Текст {j}: \"{texts[j-1][:50]}...\"\n")