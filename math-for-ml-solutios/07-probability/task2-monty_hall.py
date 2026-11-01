import numpy as np 

num_games = 100_000

wins_stay = 0
wins_switch = 0

for _ in range(num_games):
    car_door = np.random.randint(0, 3)

    player_choice = np.random.randint(0, 3)

    if player_choice == car_door:
        wins_stay += 1

    if player_choice != car_door:
        wins_switch += 1

prob_stay = wins_stay / num_games
prob_switch = wins_switch / num_games

print(f"Всего игр: {num_games}")
print(f"Побед при стратегии 'НЕ МЕНЯТЬ': {wins_stay} ({prob_stay*100:.2f}%)")
print(f"Побед при стратегии 'МЕНЯТЬ':     {wins_switch} ({prob_switch*100:.2f}%)")