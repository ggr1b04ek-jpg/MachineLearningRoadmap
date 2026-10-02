def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a ** 2 for a in v1) ** 0.5
    norm_v2 = sum(b ** 2 for b in v2) ** 0.5

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)

test_embeddings = {
    'king':     [0.56,  0.32, -0.15,  0.48,  0.21, -0.33,  0.67,  0.12, -0.28,  0.45],
    'queen':    [0.52,  0.29, -0.18,  0.44,  0.25, -0.31,  0.63,  0.15, -0.24,  0.41],
    'man':      [0.42,  0.18, -0.08,  0.35,  0.12, -0.22,  0.48,  0.09, -0.19,  0.31],
    'woman':    [0.38,  0.15, -0.11,  0.31,  0.16, -0.20,  0.44,  0.12, -0.15,  0.27],
    'prince':   [0.49,  0.28, -0.12,  0.42,  0.18, -0.29,  0.58,  0.11, -0.25,  0.39],
    'princess': [0.45,  0.25, -0.14,  0.38,  0.22, -0.27,  0.54,  0.14, -0.21,  0.35],
    'boy':      [0.35,  0.12, -0.05,  0.28,  0.08, -0.18,  0.39,  0.06, -0.14,  0.24],
    'girl':     [0.32,  0.10, -0.07,  0.25,  0.11, -0.16,  0.36,  0.09, -0.11,  0.21],
    'actor':    [0.41,  0.22, -0.09,  0.36,  0.14, -0.24,  0.47,  0.10, -0.20,  0.33],
    'actress':  [0.37,  0.19, -0.12,  0.32,  0.18, -0.22,  0.43,  0.13, -0.16,  0.29],
    'father':   [0.44,  0.20, -0.10,  0.38,  0.15, -0.25,  0.51,  0.08, -0.22,  0.35],
    'mother':   [0.40,  0.17, -0.13,  0.34,  0.19, -0.23,  0.47,  0.11, -0.18,  0.31],
    'emperor':  [0.58,  0.34, -0.16,  0.50,  0.23, -0.35,  0.69,  0.13, -0.30,  0.47],
    'empress':  [0.54,  0.31, -0.19,  0.46,  0.27, -0.33,  0.65,  0.16, -0.26,  0.43],
    'lord':     [0.47,  0.26, -0.11,  0.40,  0.17, -0.28,  0.56,  0.10, -0.23,  0.37],
    'lady':     [0.43,  0.23, -0.14,  0.36,  0.21, -0.26,  0.52,  0.13, -0.19,  0.33],
    'dog':      [0.15, -0.08,  0.22,  0.11, -0.05,  0.18,  0.19, -0.12,  0.25,  0.09],
    'cat':      [0.13, -0.10,  0.24,  0.09, -0.07,  0.16,  0.17, -0.14,  0.27,  0.07],
    'car':      [-0.05,  0.28,  0.15, -0.12,  0.32,  0.08, -0.18,  0.25,  0.11, -0.22],
    'house':    [0.08,  0.25,  0.18, -0.09,  0.29,  0.11, -0.15,  0.22,  0.14, -0.19],
}

def find_most_similar(word, embeddings, top_n=5, exclude=None):
    if word not in embeddings:
        print(f"Слово {word} не найдено")
        return []
    
    if exclude is None:
        exclude = set()
    
    word_vec = embeddings[word]
    similarities = []

    for vocab_word, vec in embeddings.items():
        if vocab_word == word or vocab_word in exclude:
            continue

        sim = cosine_similarity(word_vec, vec)
        similarities.append((vocab_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]

print("Топ-5 похожих слов для 'king':")
similar_to_king = find_most_similar('king', test_embeddings, top_n=5)

for i, (word, score) in enumerate(similar_to_king, 1):
    print(f"{i}. {word:12s}: {score:.4f}")