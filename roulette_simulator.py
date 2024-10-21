import random
from collections import Counter

class Roulette:
    def __init__(self):
        self.wheel = [i for i in range(37)]  # 0-36 for European roulette
        self.colors = {
            0: 'green',
            1: 'red', 2: 'black', 3: 'red', 4: 'black', 5: 'red', 6: 'black', 7: 'red', 8: 'black', 9: 'red', 10: 'black',
            11: 'black', 12: 'red', 13: 'black', 14: 'red', 15: 'black', 16: 'red', 17: 'black', 18: 'red',
            19: 'red', 20: 'black', 21: 'red', 22: 'black', 23: 'red', 24: 'black', 25: 'red', 26: 'black', 27: 'red', 28: 'black',
            29: 'black', 30: 'red', 31: 'black', 32: 'red', 33: 'black', 34: 'red', 35: 'black', 36: 'red'
        }
        self.history = []

    def spin(self):
        number = random.choice(self.wheel)
        color = self.colors[number]
        self.history.append((number, color))
        return number, color

    def get_history(self):
        return self.history

class AmericanRoulette(Roulette):
    def __init__(self):
        super().__init__()
        self.wheel.append(37)  # Reprezentând 00
        self.colors[37] = 'green'  # 00 este verde

class BettingStrategy:
    def __init__(self, initial_bet, balance):
        self.initial_bet = initial_bet
        self.current_bet = initial_bet
        self.balance = balance
        self.balance_history = []
        self.spin_history = []
        self.roulette = Roulette()  # Creăm o instanță a clasei Roulette

    def apply_strategy(self, spins, strategy='martingale'):
        for _ in range(spins):
            number, color = self.roulette.spin()
            self.spin_history.append((number, color))
            outcome, payout = self.place_bet('red', self.current_bet)
            self.balance += payout - self.current_bet
            self.balance_history.append(self.balance)
            if strategy == 'martingale':
                self.current_bet = self.initial_bet if outcome == "Won" else self.current_bet * 2
            if self.balance <= 0:
                return f"Bankrupt after {len(self.balance_history)} spins."
        return f"Balance after {spins} spins: {self.balance}"

    def place_bet(self, bet, amount):
        number, color = self.roulette.spin()
        if (bet == 'red' and color == 'red') or (bet == 'black' and color == 'black'):
            return "Won", amount * 2
        elif bet == 'green' and (number == 0 or number == 37):  # 37 reprezintă 00
            return "Won", amount * 35
        else:
            return "Lost", 0

    def get_spin_history(self):
        return self.spin_history

    def get_balance_history(self):
        return self.balance_history

class RouletteStats:
    @staticmethod
    def number_frequency(history):
        numbers = [spin[0] for spin in history]  # Extragem doar numerele din istoric
        return dict(Counter(numbers))

    @staticmethod
    def color_distribution(history):
        colors = [spin[1] for spin in history]  # Extragem doar culorile din istoric
        return dict(Counter(colors))

    @staticmethod
    def hot_and_cold_numbers(history, top_n=5):
        number_counts = Counter([spin[0] for spin in history])
        hot_numbers = number_counts.most_common(top_n)
        cold_numbers = sorted(number_counts.items(), key=lambda x: x[1])[:top_n]
        return hot_numbers, cold_numbers