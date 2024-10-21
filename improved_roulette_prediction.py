import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import random
import sqlite3
import json

class ImprovedRoulettePrediction:
    def __init__(self, db_path='roulette_data.db'):
        self.db_path = db_path
        self.create_tables()
        self.algorithms = {
            'markov': self.markov_chain_prediction,
            'bayesian': self.bayesian_prediction,
            'machine_learning': self.machine_learning_prediction,
            'pattern_recognition': self.pattern_recognition,
            'statistical_analysis': self.statistical_analysis
        }
        self.weights = {algo: 1 for algo in self.algorithms}
        self.colors = {
            0: 'green',
            1: 'red', 2: 'black', 3: 'red', 4: 'black', 5: 'red', 6: 'black', 7: 'red', 8: 'black', 9: 'red', 10: 'black',
            11: 'black', 12: 'red', 13: 'black', 14: 'red', 15: 'black', 16: 'red', 17: 'black', 18: 'red',
            19: 'red', 20: 'black', 21: 'red', 22: 'black', 23: 'red', 24: 'black', 25: 'red', 26: 'black', 27: 'red', 28: 'black',
            29: 'black', 30: 'red', 31: 'black', 32: 'red', 33: 'black', 34: 'red', 35: 'black', 36: 'red'
        }


    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sequence TEXT,
                prediction INTEGER,
                actual INTEGER,
                algorithm TEXT,
                accuracy REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT,
                average_accuracy REAL,
                weight REAL
            )
        ''')
        conn.commit()
        conn.close()

    def markov_chain_prediction(self, sequence, order=2):
        if len(sequence) < order + 1:
            return random.choice(sequence)  # Returnăm un număr aleatoriu din secvență dacă e prea scurtă
        
        states = list(set(sequence))
        n_states = len(states)
        M = np.zeros((n_states**order, n_states))

        for i in range(len(sequence) - order):
            current = tuple(sequence[i:i+order])
            next_state = sequence[i+order]
            current_idx = sum([states.index(current[j]) * (n_states ** (order-j-1)) for j in range(order)])
            M[current_idx, states.index(next_state)] += 1

        M = M / (M.sum(axis=1, keepdims=True) + 1e-8)  # Adăugăm o valoare mică pentru a evita împărțirea la zero
        last_state = tuple(sequence[-order:])
        last_state_idx = sum([states.index(last_state[j]) * (n_states ** (order-j-1)) for j in range(order)])
        next_prob = M[last_state_idx]
        
        if np.sum(next_prob) == 0:
            return random.choice(sequence)  # Returnăm un număr aleatoriu dacă nu avem predicții
        
        prediction = states[np.argmax(next_prob)]
        return prediction

    def bayesian_prediction(self, sequence):
        prior = np.ones(37) / 37  # Flat prior
        for num in sequence:
            likelihood = np.zeros(37)
            likelihood[num] = 1
            posterior = prior * likelihood
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                posterior /= posterior_sum
            else:
                # Dacă suma este zero, resetăm la prior-ul inițial
                posterior = np.ones(37) / 37
            prior = posterior

        if np.isnan(posterior).any() or np.sum(posterior) == 0:
            # Dacă avem NaN sau suma este zero, returnăm o predicție aleatoare
            return random.randint(0, 36)
        
        return np.argmax(posterior)


    def load_data_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT number_sequence FROM sequences')
        sequences = cursor.fetchall()
        conn.close()
        
        all_data = []
        for seq in sequences:
            numbers = [int(num) for num in seq[0].split(',')]
            all_data.extend(numbers)
        return all_data

    def prepare_ml_data(self, data, window_size=5):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    def train_ml_model(self):
        data = self.load_data_from_db()
        if len(data) < 100:  # Verificăm dacă avem suficiente date pentru antrenament
            print("Nu sunt suficiente date pentru antrenarea modelului ML.")
            return
        
        X, y = self.prepare_ml_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        score = self.ml_model.score(X_test, y_test)
        print(f"Scorul modelului ML pe datele de test: {score}")

    def machine_learning_prediction(self, sequence):
        if self.ml_model is None or len(sequence) < 5:
            return random.choice(sequence) if sequence else 0
        
        recent_sequence = np.array(sequence[-5:]).reshape(1, -1)
        prediction = self.ml_model.predict(recent_sequence)
        return int(round(prediction[0]))


    def pattern_recognition(self, sequence):
        patterns = {}
        for i in range(len(sequence) - 5):
            pattern = tuple(sequence[i:i+5])
            if pattern in patterns:
                patterns[pattern].append(sequence[i+5])
            else:
                patterns[pattern] = [sequence[i+5]]
        
        last_pattern = tuple(sequence[-5:])
        if last_pattern in patterns:
            predictions = patterns[last_pattern]
            return max(set(predictions), key=predictions.count)
        else:
            return sequence[-1]  # Return last number if no pattern found

    def statistical_analysis(self, sequence):
        counts = Counter(sequence)
        if len(counts) < 2:
            return sequence[0]  # Returnăm singurul număr din secvență dacă e prea scurtă
        
        try:
            chi2, p_value, _, _ = chi2_contingency([list(counts.values())])
            if p_value < 0.05:  # Deviere semnificativă de la distribuția uniformă
                return min(counts, key=counts.get)  # Predicem numărul cel mai puțin frecvent
            else:
                return random.choice(sequence)  # Predicție aleatorie dacă e uniformă
        except ValueError:
            return random.choice(sequence)  # Returnam o predicție aleatorie în caz de eroare


    def combined_prediction(self, sequence):
        predictions = {}
        for algo, func in self.algorithms.items():
            try:
                predictions[algo] = func(sequence)
            except Exception as e:
                print(f"Error in {algo} prediction: {str(e)}")
                predictions[algo] = random.choice(sequence)  # Folosim o predicție aleatorie ca fallback
        
        weighted_predictions = [pred * self.weights[algo] for algo, pred in predictions.items()]
        final_prediction = int(round(sum(weighted_predictions) / sum(self.weights.values())))
        return final_prediction

    def update_weights(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for algo in self.algorithms:
            cursor.execute('''
                SELECT AVG(accuracy) FROM predictions
                WHERE algorithm = ? AND timestamp > datetime('now', '-1 day')
            ''', (algo,))
            avg_accuracy = cursor.fetchone()[0] or 0
            self.weights[algo] = max(0.1, avg_accuracy)  # Minimum weight of 0.1
            cursor.execute('''
                INSERT OR REPLACE INTO algorithm_performance (algorithm, average_accuracy, weight)
                VALUES (?, ?, ?)
            ''', (algo, avg_accuracy, self.weights[algo]))
        conn.commit()
        conn.close()

    def save_prediction(self, sequence, prediction, actual, algorithm, accuracy):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (sequence, prediction, actual, algorithm, accuracy)
            VALUES (?, ?, ?, ?, ?)
        ''', (json.dumps(sequence), prediction, actual, algorithm, accuracy))
        conn.commit()
        conn.close()

    def analyze_spin_history(self, spin_history):
        numbers, colors = zip(*spin_history)
        
        number_frequency = Counter(numbers)
        color_distribution = Counter(colors)
        
        hot_numbers = number_frequency.most_common(3)
        cold_numbers = sorted(number_frequency.items(), key=lambda x: x[1])[:3]
        
        return {
            'number_frequency': dict(number_frequency),
            'color_distribution': dict(color_distribution),
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers
        }

    def predict_next_number(self, sequence):
        if len(sequence) < 6:
            return random.choice(sequence) if sequence else 0
        
        predictions = {}
        for algo, func in self.algorithms.items():
            try:
                predictions[algo] = func(sequence)
            except Exception as e:
                print(f"Error in {algo} prediction: {str(e)}")
                predictions[algo] = random.choice(sequence)
        
        weighted_predictions = [pred * self.weights[algo] for algo, pred in predictions.items()]
        final_prediction = int(round(sum(weighted_predictions) / sum(self.weights.values())))
        return final_prediction

    def get_color(self, number):
        return self.colors.get(number, 'unknown')

    def get_all_predictions(self, sequence):
        if len(sequence) < 6:
            return {}
        
        predictions = {}
        for algo, func in self.algorithms.items():
            try:
                predictions[algo] = func(sequence)
            except Exception as e:
                print(f"Error in {algo} prediction: {str(e)}")
                predictions[algo] = random.choice(sequence)
        
        return predictions

    def evaluate_prediction(self, sequence, prediction, actual):
        accuracy = 1 if prediction == actual else 0
        for algo, func in self.algorithms.items():
            algo_prediction = func(sequence)
            algo_accuracy = 1 if algo_prediction == actual else 0
            self.save_prediction(sequence, algo_prediction, actual, algo, algo_accuracy)
        
        self.update_weights()
        return accuracy