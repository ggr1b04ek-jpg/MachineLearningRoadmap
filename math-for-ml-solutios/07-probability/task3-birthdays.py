import numpy as np 

def analytical_prob(n):
    if n > 365:
        return 1.0
    p_unique = 1.0
    for i in range(n):
        p_unique *= (365 - i) / 365
    return 1 - p_unique

def simulate_prob(n, num_simulations=10_000):
    matches = 0
    for _ in range(num_simulations):
        birthdays = np.random.randint(0, 365, size=n)
        if len(np.unique(birthdays)) != birthdays.size:
            matches += 1
    
    return matches / num_simulations

group_sizes = np.arange(1, 61)

analytical_probs = []
simulated_probs = []

print(f"{'Размер группы':<15} | {'Аналитика':<10} | {'Симуляция':<10}")
print("-" * 45)

for n in group_sizes:
    p_anal = analytical_prob(n)
    p_sim = simulate_prob(n, num_simulations=5_000)
    
    analytical_probs.append(p_anal)
    simulated_probs.append(p_sim)
    
    if n in [20, 23, 30, 50]:
        print(f"{n:<15} | {p_anal:<10.4f} | {p_sim:<10.4f}")