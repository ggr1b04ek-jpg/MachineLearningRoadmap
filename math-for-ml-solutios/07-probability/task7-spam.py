import numpy as np 

texts = [
    "free money now",
    "win lottery prize",
    "hello friend",
    "meeting tomorrow",
    "click here free"
]
labels = [1, 1, 0, 0, 1]

vocabulary = set()
for text in texts:
    words = text.lower().split()
    vocabulary.update(words)
vocabulary = sorted(list(vocabulary))
print("Словарь:", vocabulary)
print("Размер словаря:", len(vocabulary))

X = np.zeros((len(texts), len(vocabulary)))
for i, text in enumerate(texts):
    words = text.lower().split()
    for word in words:
        j = vocabulary.index(word)
        X[i, j] = 1

print("\nМатрица признаков:")
print(X)
print("Метки:", labels)

num_spam = sum(labels)
total = len(labels)
num_not_spam = total - num_spam

p_spam = num_spam / total
p_not_spam = num_not_spam / total

num_words = len(vocabulary)
p_word_given_spam = np.zeros(num_words)
p_word_given_not_spam = np.zeros(num_words)

for j in range(num_words):
    spam_count = sum(X[i, j] for i in range(total) if labels[i] == 1)
    not_spam_count = sum(X[i, j] for i in range(total) if labels[i] == 0)

    p_word_given_spam[j] = (spam_count + 1) / (num_spam + 2)
    p_word_given_not_spam[j] = (not_spam_count + 1) / (num_not_spam + 2)

for j, word in enumerate(vocabulary):
    print(f"  {word}: {p_word_given_spam[j]:.4f}")

def predict(text):
    words = text.lower().split()
    
    log_prob_spam = np.log(p_spam)
    log_prob_not_spam = np.log(p_not_spam)
    
    for word in words:
        if word in vocabulary:
            j = vocabulary.index(word)
            log_prob_spam += np.log(p_word_given_spam[j])
            log_prob_not_spam += np.log(p_word_given_not_spam[j])
    
    if log_prob_spam > log_prob_not_spam:
        return 1, log_prob_spam, log_prob_not_spam
    else:
        return 0, log_prob_spam, log_prob_not_spam

test_texts = [
    "free prize",
    "hello tomorrow",
    "win money",
    "friend meeting"
]

print("\n" + "="*60)
print("ТЕСТИРОВАНИЕ")
print("="*60)
for text in test_texts:
    pred, log_spam, log_not_spam = predict(text)
    label = "СПАМ" if pred == 1 else "НЕ СПАМ"
    print(f"\nТекст: '{text}'")
    print(f"Предсказание: {label}")
    print(f"Log p(spam) = {log_spam:.4f}, Log p(not spam) = {log_not_spam:.4f}")