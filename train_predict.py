import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def prepare_ball_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()

    # Convert data to a list of floats
    data = [float(line.strip()) for line in data]

    # Group data into pairs (x, y)
    positions = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

    # Convert to NumPy array
    training_data = np.array(positions)

    return training_data


def train_lstm_model(training_data):
    # Assume training_data is a NumPy array of shape (num_samples, 2)
    sequence_length = 10
    X_train = []
    y_train = []

    # Prepare sequences
    for i in range(len(training_data) - sequence_length):
        X_train.append(training_data[i : i + sequence_length])
        y_train.append(training_data[i + sequence_length])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(sequence_length, 2)))
    model.add(Dense(2, activation="relu"))  # Use 'relu' to ensure non-negative outputs

    # Compile and train the model
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save("ball_position_model.keras")
    return model


training_data = prepare_ball_data("path_to_ball_positions.txt")
model = train_lstm_model(training_data)
