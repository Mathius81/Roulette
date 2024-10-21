from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sqlite3
import csv
import json
from roulette_analysis import analyze_spin_history, analyze_color_sequences, predict_next_number
from roulette_simulator import BettingStrategy, Roulette
from collections import Counter
from random import choice
from improved_roulette_prediction import ImprovedRoulettePrediction
from gpt4o_integration import get_gpt4o_prediction
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

live_feed_data = []
roulette = Roulette() 
roulette_predictor = ImprovedRoulettePrediction()

last_prediction = None
last_prediction_accuracy = None

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def connect_db():
    conn = sqlite3.connect('roulette_data.db')
    return conn, conn.cursor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_db():
    conn, cursor = connect_db()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number_sequence TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_sequence(cursor, sequence):
    cursor.execute('INSERT INTO sequences (number_sequence) VALUES (?)', (','.join(map(str, sequence)),))

def process_file(file_path):
    conn, cursor = connect_db()

    if file_path.endswith('.csv'):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                sequence = list(map(int, row))
                insert_sequence(cursor, sequence)

    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            content = file.read().strip()
            lines = content.split(',')
            numbers = []
            
            for item in lines:
                try:
                    number = int(item.strip())
                    numbers.append(number)
                except ValueError:
                    print(f"Element invalid găsit și ignorat: {item}")

            sequence_size = 6
            sequences = [numbers[i:i + sequence_size] for i in range(0, len(numbers), sequence_size)]

            for sequence in sequences:
                if len(sequence) == sequence_size:
                    insert_sequence(cursor, sequence)

    conn.commit()
    conn.close()

@app.route('/upload_sequences', methods=['GET', 'POST'])
def upload_sequences():
    if request.method == 'POST':
        if 'sequence_file' not in request.files:
            return redirect(request.url)
        file = request.files['sequence_file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            process_file(file_path)
            roulette_predictor.train_ml_model()  
            return redirect(url_for('upload_sequences'))
    return render_template('upload.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    bet_amount = int(request.form['bet_amount'])
    strategy = request.form['strategy']
    spins = int(request.form['spins'])

    strategy_simulator = BettingStrategy(initial_bet=bet_amount, balance=1000)
    result = strategy_simulator.apply_strategy(spins, strategy=strategy)

    spin_history = strategy_simulator.get_spin_history()
    balance_history = strategy_simulator.get_balance_history()

    analysis = analyze_spin_history(spin_history)
    prediction = predict_next_number(spin_history)
    color_sequences = analyze_color_sequences(spin_history)

    return render_template('results.html', 
                           result=result, 
                           number_freq=analysis['number_frequency'],
                           color_dist=analysis['color_distribution'],
                           hot_numbers=analysis['hot_numbers'],
                           cold_numbers=analysis['cold_numbers'],
                           prediction=prediction,
                           balance_history=balance_history,
                           color_sequences=color_sequences)

def find_similar_sequences(live_sequence, window_size=6):
    conn, cursor = connect_db()
    query = 'SELECT number_sequence FROM sequences'
    cursor.execute(query)
    all_sequences = cursor.fetchall()

    best_match = None
    best_match_diff = float('inf')

    for seq in all_sequences:
        stored_seq = list(map(int, seq[0].split(',')))
        if len(stored_seq) >= window_size:
            stored_window = stored_seq[-window_size:]
            diff = sum([abs(live_sequence[i] - stored_window[i]) for i in range(window_size)])
            if diff < best_match_diff:
                best_match_diff = diff
                best_match = stored_seq

    conn.close()
    return best_match


@app.route('/live_feed', methods=['POST'])
def live_feed():
    global live_feed_data, last_prediction, last_prediction_accuracy
    
    live_number = int(request.form['live_number'])
    color = roulette_predictor.get_color(live_number)
    live_feed_data.append((live_number, color))

    sequence = [num for num, _ in live_feed_data]
    
    # Evaluarea predicției anterioare
    if last_prediction is not None:
        last_prediction_accuracy = 1 if last_prediction == live_number else 0
    
    # Generarea noii predicții
    prediction = roulette_predictor.predict_next_number(sequence)
    last_prediction = prediction

    # Obținerea predicției GPT-4o
    gpt4o_prediction = None
    if len(sequence) >= 6:
        all_predictions = roulette_predictor.get_all_predictions(sequence)
        serializable_predictions = {k: convert_numpy_types(v) for k, v in all_predictions.items()}
        # gpt4o_prediction = get_gpt4o_prediction(
        #     json.dumps(sequence),
        #     json.dumps(serializable_predictions)
        # )

    # Analiza se face pe toate numerele disponibile
    analysis = roulette_predictor.analyze_spin_history(live_feed_data)

    return render_template('live_feed.html', 
                           live_feed=live_feed_data,
                           number_freq=analysis['number_frequency'],
                           color_dist=analysis['color_distribution'],
                           hot_numbers=analysis['hot_numbers'],
                           cold_numbers=analysis['cold_numbers'],
                           prediction=prediction,
                           prediction_color=roulette_predictor.get_color(prediction),
                           prediction_accuracy=last_prediction_accuracy)


@app.route('/analyze_performance')
def analyze_performance():
    performance_data, improvement_suggestion = roulette_predictor.analyze_performance()
    return render_template('performance.html', 
                           performance_data=performance_data, 
                           improvement_suggestion=improvement_suggestion)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    create_db()
    app.run(host="0.0.0.0", port=36453, debug=True)