from collections import Counter
import random

def analyze_spin_history(spin_history, window_size=6):
    recent_spins = spin_history[-window_size:]
    numbers, colors = zip(*recent_spins)
    
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

def predict_next_spin(spin_history, last_n=10):
    recent_spins = spin_history[-last_n:]
    recent_numbers, _ = zip(*recent_spins)
    predicted_number = Counter(recent_numbers).most_common(1)[0][0]
    return predicted_number

def analyze_color_sequences(spin_history, sequence_length=3):
    colors = [color for _, color in spin_history]
    sequences = [''.join(colors[i:i+sequence_length]) for i in range(len(colors)-sequence_length+1)]
    return Counter(sequences)

def predict_next_number(live_sequence, cursor, full_live_feed):
    if len(full_live_feed) < 6:
        return full_live_feed[-1][0], "insufficient data"

    live_seq_str = ','.join(map(str, [num for num, _ in live_sequence[-6:]]))
    query = "SELECT number_sequence FROM sequences WHERE number_sequence LIKE ?"
    cursor.execute(query, (f'%{live_seq_str}%',))
    similar_sequences = cursor.fetchall()
    
    if similar_sequences:
        next_numbers = []
        for seq in similar_sequences:
            seq_list = list(map(int, seq[0].split(',')))
            try:
                start_index = seq_list.index(live_sequence[-1][0])
                if start_index < len(seq_list) - 1:
                    next_numbers.append(seq_list[start_index + 1])
            except ValueError:
                continue
        
        if next_numbers:
            prediction = max(set(next_numbers), key=next_numbers.count)
            return max(0, min(36, prediction)), "database"
    
    prediction, source = advanced_prediction([num for num, _ in full_live_feed], cursor)
    return max(0, min(36, prediction)), source