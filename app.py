from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sqlite3
import csv
from roulette_analysis import analyze_spin_history, predict_next_spin, analyze_color_sequences, predict_next_number
from roulette_simulator import BettingStrategy, Roulette
from collections import Counter
from random import choice

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize an empty list to hold the live feed data
live_feed_data = []

roulette = Roulette() 

# Conectarea la baza de date SQLite
def connect_db():
    conn = sqlite3.connect('roulette_data.db')
    return conn, conn.cursor()

# Funcție pentru a verifica extensia fișierului
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Crearea bazei de date pentru secvențe
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

# Funcție pentru a insera secvențele în baza de date
def insert_sequence(cursor, sequence):
    cursor.execute('INSERT INTO sequences (number_sequence) VALUES (?)', (','.join(map(str, sequence)),))

# Funcție pentru prelucrarea fișierelor și stocarea secvențelor
def process_file(file_path):
    conn, cursor = connect_db()

    # Procesarea fișierelor CSV
    if file_path.endswith('.csv'):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                sequence = list(map(int, row))  # Transformăm secvența în int
                insert_sequence(cursor, sequence)

    # Procesarea fișierelor TXT
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            content = file.read().strip()
            lines = content.split(',')  # Separăm fiecare element prin virgulă
            numbers = []
            
            for item in lines:
                try:
                    # Încercăm să convertim fiecare element în număr
                    number = int(item.strip())
                    numbers.append(number)
                except ValueError:
                    # Dacă nu este un număr valid, îl ignorăm
                    print(f"Element invalid găsit și ignorat: {item}")

            # Împărțim lista de numere în secvențe de câte 6 numere consecutive
            sequence_size = 6
            sequences = [numbers[i:i + sequence_size] for i in range(0, len(numbers), sequence_size)]

            for sequence in sequences:
                if len(sequence) == sequence_size:
                    insert_sequence(cursor, sequence)

    conn.commit()
    conn.close()

# Funcție pentru încărcarea secvențelor (CSV/TXT)
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
    prediction = predict_next_spin(spin_history)
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

# Funcție pentru găsirea secvențelor similare
def find_similar_sequences(live_sequence, window_size=6):
    conn, cursor = connect_db()
    query = 'SELECT number_sequence FROM sequences'
    cursor.execute(query)
    all_sequences = cursor.fetchall()

    best_match = None
    best_match_diff = float('inf')  # Valoare mare pentru comparare

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

# Funcție pentru feedul live și predicția
@app.route('/live_feed', methods=['POST'])
def live_feed():
    global live_feed_data
    live_number = int(request.form['live_number'])
    color = roulette.colors[live_number]
    live_feed_data.append((live_number, color))

    conn, cursor = connect_db()
    prediction, prediction_source = predict_next_number(live_feed_data, cursor, live_feed_data)
    conn.close()

    # Asigurăm-ne că predicția este în intervalul valid
    prediction = max(0, min(36, prediction))
    prediction_color = roulette.colors[prediction]

    analysis = analyze_spin_history(live_feed_data[-6:] if len(live_feed_data) >= 6 else live_feed_data)

    return render_template('live_feed.html', 
                           live_feed=live_feed_data,
                           number_freq=analysis['number_frequency'],
                           color_dist=analysis['color_distribution'],
                           hot_numbers=analysis['hot_numbers'],
                           cold_numbers=analysis['cold_numbers'],
                           prediction=prediction,
                           prediction_color=prediction_color,
                           prediction_source=prediction_source)


# Funcție pentru pagina principală
@app.route('/')
def index():
    return render_template('index.html')

# Funcția principală
if __name__ == '__main__':
    create_db()
    app.run(host="0.0.0.0", port=36453, debug=True)