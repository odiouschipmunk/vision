import numpy as np
import clip, torch
import cv2
def find_match_2d_array(array, x):
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False


def drawmap(lx, ly, rx, ry, map):

    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), map.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), map.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), map.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), map.shape[0] - 1)
    map[ly, lx] += 1
    map[ry, rx] += 1


def get_image_embeddings(image):
    imagemodel, preprocess = clip.load("ViT-B/32", device="cpu")
    image = preprocess(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        embeddings = imagemodel.encode_image(image)
    return embeddings.cpu().numpy()


# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    # Flatten the embeddings to 1D if they are 2D (like (1, 512))
    embedding1 = np.squeeze(embedding1)  # Shape becomes (512,)
    embedding2 = np.squeeze(embedding2)  # Shape becomes (512,)

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Check if any norm is zero to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0  # Return a similarity of 0 if one of the embeddings is invalid

    return dot_product / (norm1 * norm2)


def sum_pixels_in_bbox(frame, bbox):
    x, y, w, h = bbox
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    return np.sum(roi, dtype=np.int64)


def findLastOne(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 1:
            possibleis.append(i)
    # print(possibleis)
    if len(possibleis) > 1:
        return possibleis[-1]

    return -1


def findLastTwo(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 2:
            possibleis.append(i)
    if len(possibleis) > 1:
        return possibleis[-1]
    return -1


def findLast(i, otherTrackIds):
    possibleits = []
    for it in range(len(otherTrackIds)):
        if otherTrackIds[it][1] == i:
            possibleits.append(it)
    return possibleits[-1]

def pixel_to_3d(pixel_point, pixel_reference, reference_points_3d):
    """
    Maps a single 2D pixel coordinate to a 3D position based on reference points.

    Parameters:
        pixel_point (list): Single [x, y] pixel coordinate to map.
        pixel_reference (list): List of [x, y] reference points in pixels.
        reference_points_3d (list): List of [x, y, z] reference points in 3D space.

    Returns:
        list: Mapped 3D coordinates in the form [x, y, z].
    """
    # Convert 2D reference points and 3D points to NumPy arrays
    pixel_reference_np = np.array(pixel_reference, dtype=np.float32)
    reference_points_3d_np = np.array(reference_points_3d, dtype=np.float32)

    # Extract only the x and y values from the 3D reference points for homography calculation
    reference_points_2d = reference_points_3d_np[:, :2]

    # Calculate the homography matrix from 2D pixel reference to 2D real-world reference (ignoring z)
    H, _ = cv2.findHomography(pixel_reference_np, reference_points_2d)

    # Ensure pixel_point is in homogeneous coordinates [x, y, 1]
    pixel_point_homogeneous = np.array([pixel_point[0], pixel_point[1], 1], dtype=np.float32)

    # Apply the homography matrix to get a 2D point in real-world space
    real_world_2d = np.dot(H, pixel_point_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize to make it [x, y, 1]

    # Now interpolate the z-coordinate based on distances
    # Calculate weights based on the nearest reference points in the 2D plane
    distances = np.linalg.norm(reference_points_2d - real_world_2d[:2], axis=1)
    weights = 1 / (distances + 1e-5)  # Avoid division by zero
    z_mapped = np.dot(weights, reference_points_3d_np[:, 2]) / np.sum(weights)

    # Combine the 2D mapped point with interpolated z to get the 3D position
    mapped_3d_point = [real_world_2d[0], real_world_2d[1], z_mapped]

    return mapped_3d_point



def transform_and_display(player1_points, player2_points, pixel_reference, reference_points_3d, image):
    """
    Transforms and displays the positions of Player 1 and Player 2 using homography.

    Parameters:
        player1_points (list): List of [x, y] pixel points for Player 1.
        player2_points (list): List of [x, y] pixel points for Player 2.
        pixel_reference (list): List of [x, y] reference points in pixels.
        reference_points_3d (list): List of [x, y, z] reference points in 3D space.
        image (np.array): Original image with players to transform and display.

    Returns:
        None: Displays the original and transformed images side-by-side.
    """
    # Ensure player points are in 2D format
    player1_points = np.array(player1_points, dtype=np.float32).reshape(-1, 2)
    player2_points = np.array(player2_points, dtype=np.float32).reshape(-1, 2)

    # Convert pixel_reference and reference_points_3d to numpy arrays for processing
    pixel_reference_np = np.array(pixel_reference, dtype=np.float32)
    reference_points_2d = np.array([point[:2] for point in reference_points_3d], dtype=np.float32)  # Only x, y

    # Verify that we have enough points for homography
    if pixel_reference_np.shape[0] < 4 or reference_points_2d.shape[0] < 4:
        print("Error: Not enough reference points for homography.")
        return

    # Calculate the homography matrix to map the 2D pixel reference to real-world 2D reference
    H, status = cv2.findHomography(pixel_reference_np, reference_points_2d)
    
    # Check if homography calculation was successful
    if H is None:
        print("Homography calculation failed.")
        return

    # Apply homography to player points to get normalized view
    player1_points_homogeneous = np.hstack([player1_points, np.ones((len(player1_points), 1))])
    player2_points_homogeneous = np.hstack([player2_points, np.ones((len(player2_points), 1))])

    # Transform the points for each player
    transformed_player1 = (H @ player1_points_homogeneous.T).T
    transformed_player2 = (H @ player2_points_homogeneous.T).T

    # Normalize by dividing by the third coordinate to get (x, y, 1)
    transformed_player1 = transformed_player1[:, :2] / transformed_player1[:, 2][:, None]
    transformed_player2 = transformed_player2[:, :2] / transformed_player2[:, 2][:, None]

    # Draw original and transformed points on separate images for side-by-side comparison
    image_original = image.copy()
    
    # Set destination image size explicitly to prevent black screen
    h, w = image.shape[:2]
    image_transformed = cv2.warpPerspective(image, H, (w, h))

    # Draw original points on the original image
    for (x, y) in player1_points:
        cv2.circle(image_original, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blue for Player 1
    for (x, y) in player2_points:
        cv2.circle(image_original, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for Player 2

    # Draw transformed points on the transformed image
    for (x, y) in transformed_player1:
        cv2.circle(image_transformed, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blue for Player 1
    for (x, y) in transformed_player2:
        cv2.circle(image_transformed, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green for Player 2

    # Combine images side-by-side for comparison
    combined_image = np.hstack((image_original, image_transformed))

    # Display the images
    cv2.imshow("Original (Left) and Transformed (Right)", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()