import pandas as pd 
import seaborn as sns

df = sns.load_dataset('titanic')

p_survived = df['survived'].mean()

p_survived_female = df[df['sex'] == 'female']['survived'].mean()

p_survived_male = df[df['sex'] == 'male']['survived'].mean()

p_survived_1_cl = df[df['pclass'] == 1]['survived'].mean()

print(f"Базовая вероятность (Prior):      {p_survived:.1%}")
print(f"Если женщина (Posterior):         {p_survived_female:.1%} (отклонение: +{p_survived_female - p_survived:.1%})")
print(f"Если мужчина (Posterior):         {p_survived_male:.1%} (отклонение: {p_survived_male - p_survived:.1%})")
print(f"Если 1 класс (Posterior):         {p_survived_1_cl:.1%} (отклонение: +{p_survived_1_cl - p_survived:.1%})")