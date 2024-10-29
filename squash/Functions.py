import numpy as np
import clip
import torch
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




def transform_pixel_to_real_world(pixel_points, H):
    """
    Transform pixel points to real-world coordinates using the homography matrix.

    Parameters:
        pixel_points (list): List of [x, y] pixel coordinates to transform.
        H (np.array): Homography matrix.

    Returns:
        list: Transformed real-world coordinates in the form [x, y].
    """
    # Convert pixel points to homogeneous coordinates for matrix multiplication
    pixel_points_homogeneous = np.append(pixel_points, 1)

    # Apply the homography matrix to get a 2D point in real-world space
    real_world_2d = np.dot(H, pixel_points_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize

    return real_world_2d[:2]

def display_player_positions(rlworldp1, rlworldp2):
    """
    Display the player positions on another screen using OpenCV.

    Parameters:
        rlworldp1 (list): Real-world coordinates of player 1.
        rlworldp2 (list): Real-world coordinates of player 2.

    Returns:
        None
    """
    # Create a blank image
    display_image = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw player positions
    cv2.circle(display_image, (int(rlworldp1[0]), int(rlworldp1[1])), 5, (255, 0, 0), -1)  # Blue for player 1
    cv2.circle(display_image, (int(rlworldp2[0]), int(rlworldp2[1])), 5, (0, 0, 255), -1)  # Red for player 2

    # Display the image
    cv2.imshow("Player Positions", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#function to determine if an array has a false positive in the last threshold frames
def is_ball_false_pos(ball_pos, threshold=5):
    if len(ball_pos) < threshold:
        return False
    #assuming the ball_pos array is formatted as [[x1,y1,frame1],[x2,y2,frame2],...]
    #sort the array by frame number
    ball_pos.sort(key=lambda x: x[2])
    #get the last threshold positions
    thresh_pos=ball_pos[-threshold:]
    #go through each position and check if the x and y values are the same
    for i in range(1,threshold):
        for j in range(0, i):
            if i==j:
                continue
            if thresh_pos[i][0] == thresh_pos[j][0] and thresh_pos[i][1] == thresh_pos[j][1]:
                return True
    return False

def validate_reference_points(px_points, rl_points):

    """
    Validate reference points for homography calculation.
    
    Parameters:
        px_points: List of pixel coordinates [[x, y], ...]
        rl_points: List of real-world coordinates [[X, Y, Z], ...] or [[X, Y], ...]
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if len(px_points) != len(rl_points):
        return False, "Number of pixel and real-world points must match"
        
    if len(px_points) < 4:
        return False, "At least 4 point pairs are required for homography calculation"
        
    # Check pixel points format
    if not all(len(p) == 2 for p in px_points):
        return False, "Pixel points must be 2D coordinates [x, y]"
        
    # Check real-world points format
    if not all(len(p) in [2, 3] for p in rl_points):
        return False, "Real-world points must be either 2D [X, Y] or 3D [X, Y, Z] coordinates"
        
    return True, ""

#function to generate homography based on referencepoints in the video in pixel[x,y] format and also real world reference points in the form of [x,y,z] in meters
def generate_homography(px_reference_points, rl_reference_points):
    """
    Generate a homography matrix from pixel to real-world coordinates.

    Parameters:
        px_reference_points (list): List of pixel reference points [[x, y], ...].
        rl_reference_points (list): List of real-world reference points [[X, Y, Z], ...].

    Returns:
        np.array: Homography matrix that maps pixel points to real-world points.
    """
    # Convert reference points to NumPy arrays for processing
    px_reference_points_np = np.array(px_reference_points, dtype=np.float32)
    rl_reference_points_np = np.array(rl_reference_points, dtype=np.float32)

    # Use only X and Y for homography as it’s a 2D transformation
    rl_reference_points_2d = rl_reference_points_np[:, :2]

    # Compute the homography matrix
    H, _ = cv2.findHomography(px_reference_points_np, rl_reference_points_2d)

    return H

def pixel_to_3d(pixel_point, H, rl_reference_points):
    """
    Convert a pixel point to an interpolated 3D real-world point using the homography matrix.

    Parameters:
        pixel_point (list): Pixel coordinate [x, y] to transform.
        H (np.array): Homography matrix from `generate_homography`.
        rl_reference_points (list): List of real-world coordinates [[X, Y, Z], ...].

    Returns:
        list: Estimated interpolated 3D coordinate in the form [X, Y, Z].
    """
    # Convert pixel point to homogeneous coordinates
    pixel_point_homogeneous = np.array([*pixel_point, 1])

    # Map pixel point to real-world 2D using the homography matrix
    real_world_2d = np.dot(H, pixel_point_homogeneous)
    real_world_2d /= real_world_2d[2]  # Normalize to get actual coordinates

    # Convert real-world reference points to NumPy array
    rl_reference_points_np = np.array(rl_reference_points, dtype=np.float32)

    # Calculate distances in the X-Y plane
    distances = np.linalg.norm(rl_reference_points_np[:, :2] - real_world_2d[:2], axis=1)

    # Calculate weights inversely proportional to distances for interpolation
    weights = 1 / (distances + 1e-6)  # Avoid division by zero with epsilon
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Perform weighted interpolation for the X, Y, and Z coordinates
    interpolated_x = np.dot(weights, rl_reference_points_np[:, 0])
    interpolated_y = np.dot(weights, rl_reference_points_np[:, 1])
    interpolated_z = np.dot(weights, rl_reference_points_np[:, 2])

    return [round(interpolated_x, 3), round(interpolated_y, 3), round(interpolated_z, 3)]


def apply_homography(H, points, inverse=False):

    """
    Apply homography transformation to a set of points.
    
    Parameters:
        H: 3x3 homography matrix
        points: List of points to transform [[x, y], ...]
        inverse: If True, applies inverse transformation
        
    Returns:
        np.ndarray: Transformed points
    """
    try:
        points = np.array(points, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        
        if inverse:
            H = np.linalg.inv(H)
        
        # Reshape points to Nx1x2 format required by cv2.perspectiveTransform
        points_reshaped = points.reshape(-1, 1, 2)
        
        # Apply transformation
        transformed_points = cv2.perspectiveTransform(points_reshaped, H)
        
        return transformed_points.reshape(-1, 2)
        
    except Exception as e:
        raise ValueError(f"Error in apply_homography: {str(e)}")