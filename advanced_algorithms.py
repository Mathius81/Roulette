import numpy as np
from collections import Counter
from scipy.stats import dirichlet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def markov_chain_prediction(sequence, order=1):
    states = list(set(sequence))
    n_states = len(states)
    M = np.zeros((n_states, n_states))

    for i in range(len(sequence) - order):
        current = sequence[i:i+order]
        next_state = sequence[i+order]
        M[states.index(current[-1]), states.index(next_state)] += 1

    M = M / M.sum(axis=1, keepdims=True)
    last_state = sequence[-1]
    next_prob = M[states.index(last_state)]
    prediction = states[np.argmax(next_prob)]

    return prediction

def create_sequence_model(input_length):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(input_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict(sequence, input_length=6):
    if len(sequence) <= input_length:
        # Nu avem suficiente date pentru antrenare, returnăm o predicție simplă
        return sequence[-1]  # Returnăm ultimul număr ca predicție

    X, y = [], []
    for i in range(len(sequence) - input_length):
        X.append(sequence[i:i+input_length])
        y.append(sequence[i+input_length])
    
    if not X or not y:
        # Nu avem suficiente date pentru antrenare, returnăm o predicție simplă
        return sequence[-1]

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_sequence_model(input_length)
    model.fit(X, y, epochs=200, verbose=0)

    last_sequence = sequence[-input_length:]
    last_sequence = np.array(last_sequence).reshape((1, input_length, 1))
    prediction = model.predict(last_sequence)
    return round(prediction[0][0])

def find_cycles(sequence, max_length=20):
    cycles = {}
    for length in range(2, max_length + 1):
        for start in range(len(sequence) - length):
            pattern = tuple(sequence[start:start+length])
            if pattern in cycles:
                cycles[pattern].append(start)
            else:
                cycles[pattern] = [start]
    
    significant_cycles = {k: v for k, v in cycles.items() if len(v) > 1}
    return significant_cycles

def predict_from_cycles(sequence, cycles):
    for length in range(len(sequence), 1, -1):
        pattern = tuple(sequence[-length:])
        if pattern in cycles:
            next_index = (cycles[pattern][-1] + len(pattern)) % 37
            return next_index
    return None

def bayesian_update(prior, observations, numbers=range(37)):
    alpha = prior + np.bincount(observations, minlength=37)
    posterior = dirichlet(alpha).mean()
    return dict(zip(numbers, posterior))

def bayesian_prediction(sequence):
    prior = np.ones(37)  # Flat prior
    posterior = bayesian_update(prior, sequence)
    return max(posterior, key=posterior.get)

def advanced_prediction(sequence, db_cursor):
    predictions = []
    
    if len(sequence) < 6:
        return sequence[-1], "insufficient data"

    markov_pred = markov_chain_prediction(sequence)
    predictions.append(markov_pred)
    
    nn_pred = train_and_predict(sequence)
    predictions.append(nn_pred)
    
    cycles = find_cycles(sequence)
    cycle_pred = predict_from_cycles(sequence, cycles)
    if cycle_pred is not None:
        predictions.append(cycle_pred)
    
    bayes_pred = bayesian_prediction(sequence)
    predictions.append(bayes_pred)
    
    if not predictions:
        return sequence[-1], "insufficient data"
    
    final_prediction = max(set(predictions), key=predictions.count)
    
    # Asigurăm-ne că predicția este în intervalul corect (0-36)
    final_prediction = max(0, min(36, int(final_prediction)))
    
    return final_prediction, "combined advanced algorithms"