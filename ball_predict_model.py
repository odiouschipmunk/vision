import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load ball positions from the file and return a list of 2D tuples.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    # Convert the data to a list of floats
    data = [float(line.strip()) for line in data]
    
    # Group the data into pairs of coordinates (x, y)
    positions = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
    
    return positions


def prepare_dataset(positions, n_steps=10):
    """
    Prepare the dataset with sequences of `n_steps` input points
    and the next point as the target.
    """
    X = []
    y = []
    
    for i in range(len(positions) - n_steps):
        X.append(positions[i:i+n_steps])
        y.append(positions[i+n_steps])
    
    # Convert to numpy arrays and reshape for CNN input
    X = np.array(X).reshape(-1, n_steps, 2, 1)  # (num_samples, n_steps, 2, 1)
    y = np.array(y)  # (num_samples, 2)
    
    return X, y


def main():
    import time
    starttime=time.time()
    # CNN model definition
    model = Sequential()

    # Convolutional layer: input shape (n_steps, 2, 1), 32 filters, 3x1 kernel
    model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(10, 2, 1)))

    # Add another convolutional layer
    model.add(Conv2D(64, (2, 1), activation='relu'))

    # Flatten the output from the convolutional layers
    model.add(Flatten())

    # Dense layer to make the final prediction
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2))  # Output layer: 2 values (x, y)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Load ball positions from a file
    positions = load_data('output(25k)\\output\\ball-xyn.txt')
    print(positions)
    
    # Prepare the dataset
    X, y = prepare_dataset(positions)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train the CNN model
    history = model.fit(X_train, y_train, epochs=5000, validation_data=(X_val, y_val), batch_size=4)
    # Evaluate on validation set
    val_loss = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {val_loss}')
    
    model.save('ball_position_model(25k).keras') 
    # Predict the next ball position
    # Example input: the last `n_steps` positions
    input_sequence = np.array([positions[:10]])  # Example positions
    input_sequence = input_sequence.reshape((1, 10, 2, 1))  # Reshape for model input

    predicted_position = model.predict(input_sequence)
    print(f'Predicted next position: {predicted_position}')
    print(f'took {time.time()-starttime} seconds')
    



if __name__ == "__main__":
    main()