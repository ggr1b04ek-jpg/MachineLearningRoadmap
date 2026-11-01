import numpy as np 
import matplotlib.pyplot as plt 

num_tosses = 1_000_000

tosses = np.random.randint(0, 2, size=num_tosses)

cum_ratio = np.cumsum(tosses) / np.arange(1, num_tosses + 1)

plt.figure(figsize=(12, 6))

plt.plot(np.arange(1, num_tosses + 1), cum_ratio, alpha=0.7, label='Доля орлов')

plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Теоретическая вероятность (0.5)')

plt.xscale('log')
plt.xlabel('Количество бросков (лог. шкала)')
plt.ylabel('Доля выпавших орлов')
plt.title('Закон больших чисел: сходимость доли орлов к 0.5')
plt.legend()
plt.grid(True, alpha=0.3)

for x in [10, 100, 10000, 1000000]:
    plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

plt.show()

print("Количество бросков | Доля орлов")
print("-" * 30)
for n in [10, 100, 1000, 10000, 100000]:
    print(f"{n:<18} | {cum_ratio[n-1]:.4f}")