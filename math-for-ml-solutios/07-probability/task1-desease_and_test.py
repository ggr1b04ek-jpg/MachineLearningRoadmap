import numpy as np 
import pandas as pd 

prevs = [0.0001, 0.01, 0.1]
accs = [0.9, 0.99, 0.999]

results = []

for p_d in prevs:
    for acc in accs:
        p_t_given_d = acc

        p_t_given_not_d = 1 - acc

        p_t = p_d * p_t_given_d + p_t_given_not_d * (1 - p_d)

        posterior = (p_t_given_d * p_d) / p_t

        results.append({
            'Заболеваемость P(D)': f"{p_d*100:.2f}%",
            'Точность теста': f"{acc*100:.1f}%",
            'P(Болезнь | Тест+)': f"{posterior*100:.2f}%"
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))