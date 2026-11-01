import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('C:/Users/user/Downloads/ml-100k/u.data', sep='\t', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

user_ids = sorted(ratings['user_id'].unique())
movie_ids = sorted(ratings['movie_id'].unique())

user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

n_users = len(user_ids)
n_movies = len(movie_ids)

print(f"Матрица R: {n_users} пользователей × {n_movies} фильмов")
print(f"Всего оценок: {len(ratings)}")
print(f"Плотность: {len(ratings)/(n_users*n_movies)*100:.2f}%")

R_full = np.zeros((n_users, n_movies), dtype=np.float64)

for _, row in ratings.iterrows():
    u_idx = user_to_idx[row['user_id']]
    m_idx = movie_to_idx[row['movie_id']]
    R_full[u_idx, m_idx] = row['rating']

known_mask = R_full > 0
train_indices, test_indices = train_test_split(
    np.argwhere(known_mask), 
    test_size=0.2, 
    random_state=42
)

R_train = R_full.copy()
for u, m in test_indices:
    R_train[u, m] = 0.0

test_data = [(u, m, R_full[int(u), int(m)]) for u, m in test_indices]

print(f"    Train оценок: {len(train_indices)}")
print(f"    Test оценок:  {len(test_data)}\n")

k = 20
U, sigma, Vt = np.linalg.svd(R_train, full_matrices=False)

U_k = U[:, :k]
sigma_k = sigma[:k]
Vt_k = Vt[:k, :]

R_pred = U_k @ np.diag(sigma_k) @ Vt_k
print(f"    Матрица восстановлена. Размер: {R_pred.shape}")

errors = []
pred_values = []
for u, m, true_rating in test_data:
    pred_rating = R_pred[u, m]
    errors.append(np.abs(true_rating - pred_rating))
    pred_values.append(pred_rating)

mae = np.mean(errors)

print(f"    MAE: {mae}")
print(f"    Среднее предсказание: {np.mean(pred_values)}")
print(f"    Диапозон предсказаний: [{min(pred_values):.2f}, {max(pred_values):.2f}]")
