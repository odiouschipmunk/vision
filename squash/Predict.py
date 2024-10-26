import random
def predict_next_position(pos1, pos2, frame_width, frame_height, error_margin=5):
    """
    Predict the next ball position based on the past two positions with an area of error.

    Parameters:
    pos1 (tuple): The first past position (x1, y1).
    pos2 (tuple): The second past position (x2, y2).
    frame_width (int): The width of the frame.
    frame_height (int): The height of the frame.
    error_margin (int): The margin of error for the prediction.

    Returns:
    tuple: The predicted next position (x, y) with an area of error.
    """
    x1, y1 = pos1
    x2, y2 = pos2

    # Calculate the difference in positions
    dx = x2 - x1
    dy = y2 - y1

    # Predict the next position
    next_x = x2 + dx
    next_y = y2 + dy

    # Add random error within the margin
    error_x = random.randint(-error_margin, error_margin)
    error_y = random.randint(-error_margin, error_margin)

    predicted_x = next_x + error_x
    predicted_y = next_y + error_y

    # Ensure the predicted position is within the frame bounds
    predicted_x = max(0, min(predicted_x, frame_width - 1))
    predicted_y = max(0, min(predicted_y, frame_height - 1))

    return (predicted_x, predicted_y)
