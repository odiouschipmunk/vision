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
    pixel_point_homogeneous = np.array(
        [pixel_point[0], pixel_point[1], 1], dtype=np.float32
    )

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
    cv2.circle(
        display_image, (int(rlworldp1[0]), int(rlworldp1[1])), 5, (255, 0, 0), -1
    )  # Blue for player 1
    cv2.circle(
        display_image, (int(rlworldp2[0]), int(rlworldp2[1])), 5, (0, 0, 255), -1
    )  # Red for player 2

    # Display the image
    cv2.imshow("Player Positions", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# function to determine if an array has a false positive in the last threshold frames
def is_ball_false_pos(ball_pos, threshold=5):
    if len(ball_pos) < threshold:
        return False
    # assuming the ball_pos array is formatted as [[x1,y1,frame1],[x2,y2,frame2],...]
    # sort the array by frame number
    ball_pos.sort(key=lambda x: x[2])
    # get the last threshold positions
    thresh_pos = ball_pos[-threshold:]
    # go through each position and check if the x and y values are the same
    for i in range(1, threshold):
        for j in range(0, i):
            if i == j:
                continue
            if (
                thresh_pos[i][0] == thresh_pos[j][0]
                and thresh_pos[i][1] == thresh_pos[j][1]
            ):
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
        return (
            False,
            "Real-world points must be either 2D [X, Y] or 3D [X, Y, Z] coordinates",
        )

    return True, ""


# function to generate homography based on referencepoints in the video in pixel[x,y] format and also real world reference points in the form of [x,y,z] in meters
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
    distances = np.linalg.norm(
        rl_reference_points_np[:, :2] - real_world_2d[:2], axis=1
    )

    # Calculate weights inversely proportional to distances for interpolation
    weights = 1 / (distances + 1e-6)  # Avoid division by zero with epsilon
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Perform weighted interpolation for the X, Y, and Z coordinates
    interpolated_x = np.dot(weights, rl_reference_points_np[:, 0])
    interpolated_y = np.dot(weights, rl_reference_points_np[:, 1])
    interpolated_z = np.dot(weights, rl_reference_points_np[:, 2])

    return [
        round(interpolated_x, 3),
        round(interpolated_y, 3),
        round(interpolated_z, 3),
    ]


from PIL import Image
from squash import Functions
from squash.Player import Player


def framepose(
    pose_model,
    frame,
    otherTrackIds,
    updated,
    references1,
    references2,
    pixdiffs,
    players,
    frame_count,
    player_last_positions,
    frame_width,
    frame_height,
    annotated_frame,
    max_players=2,
):
    track_results = pose_model.track(frame, persist=True, show=False)

    if (
        track_results
        and hasattr(track_results[0], "keypoints")
        and track_results[0].keypoints is not None
    ):
        # Extract boxes, track IDs, and keypoints from pose results
        boxes = track_results[0].boxes.xywh.cpu()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()
        keypoints = track_results[0].keypoints.cpu().numpy()

        set(track_ids)

        # Update or add players for currently visible track IDs
        # note that this only works with occluded players < 2, still working on it :(

        for box, track_id, kp in zip(boxes, track_ids, keypoints):
            x, y, w, h = box
            player_crop = frame[int(y) : int(y + h), int(x) : int(x + w)]
            Image.fromarray(player_crop)
            # embeddings=get_image_embeddings(player_image)
            Functions.sum_pixels_in_bbox(frame, [x, y, w, h])
            if not Functions.find_match_2d_array(otherTrackIds, track_id):
                # player 1 has been updated last
                if updated[0][1] > updated[1][1]:
                    if len(references2) > 1:
                        # comparing it to itself, if percentage is greater than 75, then its probably a different player
                        # if (100 * abs(psum - references2[-1]) / psum) > 75:
                        #     otherTrackIds.append([track_id, 1])
                        #     print(
                        #         f"added track id {track_id} to player 1 using image references, as image similarity was {100*abs(psum-references2[-1])/psum}"
                        #     )
                        # else:
                        otherTrackIds.append([track_id, 2])
                        print(f"added track id {track_id} to player 2")
                else:
                    # if (100 * abs(psum - references1[-1]) / psum) > 75:
                    #     otherTrackIds.append([track_id, 2])
                    #     print(
                    #         f"added track id {track_id} to player 2 using image references, as image similarity was {100*abs(psum-references1[-1])/psum}"
                    #     )
                    # else:
                    otherTrackIds.append([track_id, 1])
                    print(f"added track id {track_id} to player 1")

            """
            not updated with otherTrackIds
            if track_ids[track_id]>2:
                print(f'track id is greater than 2: {track_ids[track_id]}')
                if track_ids[track_id] not in occluded_players:
                    occ_id=occluded_players.pop()

                    print(' occ id part 153 occluded player reassigned to another player that was occluded previously. this only works with <2 occluded players, fix this soon!!!!!')
                if len(occluded_players)==1:
                    players[occluded_players.pop()]=players[track_id.get(track_id)]
                    print(' line 156 occluded player reassigned to another player that was occluded previously. this only works with <2 occluded players, fix this soon!!!!!')
            """
            # if updated[0], then that means that player 1 was updated last
            # bc of this, we can assume that the next player is player 2
            if track_id == 1:
                playerid = 1
            elif track_id == 2:
                playerid = 2
            # updated [0] is player 1, updated [1] is player 2
            # if player1 was updated last, then player 2 is next
            # if player 2 was updated last, then player 1 is next
            # if both were updated at the same time, then player 1 is next as track ids go from 1 --> 2 im really hoping
            elif updated[0][1] > updated[1][1]:
                # if len(references2) > 1:
                #     # comparing it to itself, if percentage is greater than 75, then its probably a different player
                #     if (100 * abs(psum - references2[-1]) / psum) > 75:
                #         playerid = 1
                #     else:
                #         playerid = 2
                # else:
                playerid = 2
                # player 1 was updated last
            elif updated[0][1] < updated[1][1]:
                # if len(references1) > 1:
                #     # comparing it to itself, if percentage is greater than 75, then its probably a different player
                #     if (100 * abs(psum - references1[-1]) / psum) > 75:
                #         playerid = 2
                #     else:
                #         playerid = 1
                # else:
                playerid = 1
                # player 2 was updated last
            elif updated[0][1] == updated[1][1]:
                # if len(references1) > 1 and len(references2) > 1:
                #     if (100 * abs(psum - references1[-1]) / psum) > (
                #         100 * abs(psum - references2[-1]) / psum
                #     ):
                #         playerid = 2
                #     else:
                #         playerid = 1
                # else:
                playerid = 1
                # both players were updated at the same time, so we are assuming that player 1 is the next player
            else:
                print(f"could not find player id for track id {track_id}")
                continue

            # player reference appending for maybe other stuff
            # using track_id and not playerid so that it is definitely the correct player
            # maybe use playerid instead of track_id later on, but for right now its fine tbh
            if track_id == 1:
                references1.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
                references1[-1]
                Functions.sum_pixels_in_bbox(frame, [x, y, w, h])

                # p1embeddings.append(embeddings)

                if len(references1) > 1 and len(references2) > 1:
                    if len(pixdiffs) < 5:
                        pixdiffs.append(abs(references1[-1] - references2[-1]))
                    """
                    USE FOR COSINE SIMILARITY BOOKMARK
                    else:
                        if abs(references1[-1]-references2[-1])>2*sum(pixdiffs)/len(pixdiffs):
                            print(f'probably too big of a difference between the two players, pix diff: {abs(references1[-1]-references2[-1])} with percentage as {100*abs(references1[-1]-references2[-1])/references1[-1]}')
                        else:
                            print(f'pix diff: {abs(references1[-1]-references2[-1])}')
                            print(f'average pixel diff: {sum(pixdiffs)/(len(pixdiffs))}')
                            pixdiff1percentage.append(100*abs(references1[-1]-references2[-1])/references1[-1])
                            print(f'pixel diff in percentage for p1: {pixdiff1percentage[-1]}')
                            print(f'largest percentage pixel diff: {max(pixdiff1percentage)}')
                            print(f'smallest percentage pixel diff: {min(pixdiff1percentage)}')
                    """

                # if player1imagereference is None:
                #     player1imagereference = player_image
                # player1referenceembeddings=embeddings
                # bookmark for pixel differences and cosine similarity
                """
                if len(p2embeddings) > 1:
                    p1reference=cosine_similarity(p1embeddings[-1], player1referenceembeddings)
                    p1top2=cosine_similarity(p1embeddings[-1], p2embeddings[-1])
                    cosinediffs.append(abs(p1reference-p1top2))
                    avgcosinediff=sum(cosinediffs)/len(cosinediffs)
                    
                    print(f'average cosine difference: {avgcosinediff}')       
                    print(f'highest cosine diff: {max(cosinediffs)}') 
                    print(f'lowest cosine diff: {min(cosinediffs)}')
                    print(f'this is player 1, with a cosine similarity of {p1reference} to its reference image')
                    print(f'this is player 1, with a cosine similarity of {p1top2} to player 2 right now')
                    print(f'difference between p1 reference and p2 right now: {abs(p1reference-p1top2)}')
                    """

            elif track_id == 2:
                references2.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
                references2[-1]
                Functions.sum_pixels_in_bbox(frame, [x, y, w, h])
                # print(f'p2ref: {p2ref}')
                # print(embeddings.shape)
                # p2embeddings.append(embeddings)

                if len(references1) > 1 and len(references2) > 1:
                    if len(pixdiffs) < 5:
                        pixdiffs.append(abs(references1[-1] - references2[-1]))
                    """
                    USE FOR COSINE SIMILARITY BOOKMARK
                    else:
                        if abs(references1[-1]-references2[-1])>2*sum(pixdiffs)/len(pixdiffs):
                            print(f'probably too big of a difference between the two players with pix diff: {abs(references1[-1]-references2[-1])} with percentage as {100*abs(references2[-1]-references1[-1])/references2[-1]}')
                        else:
                            print(f'pix diff: {abs(references1[-1]-references2[-1])}')
                            print(f'average pixel diff: {sum(pixdiffs)/(len(pixdiffs))}')
                            pixdiff1percentage.append(100*abs(references2[-1]-references1[-1])/references2[-1])
                            print(f'pixel diff in percentage for p2: {pixdiff2percentage[-1]}')
                            print(f'largest percentage pixel diff: {max(pixdiff2percentage)}')
                            print(f'smallest percentage pixel diff: {min(pixdiff2percentage)}')
                    """

                # print(p2embeddings)
                """
                if player2imagereference is None:
                    player2imagereference=player_image
                    player2referenceembeddings=embeddings
                


                if len(p1embeddings) > 1:
                    p2reference=cosine_similarity(p2embeddings[-1], player2referenceembeddings)
                    p2top1=cosine_similarity(p2embeddings[-1], p1embeddings[-1])
                    
                    cosinediffs.append(abs(p2reference-p2top1))
                    avgcosinediff=sum(cosinediffs)/len(cosinediffs)
                    
                    print(f'this is player 2, with a cosine similarity of {p2reference} to its reference image')
                    print(f'this is player 2, with a cosine similarity of {p2top1} to player 1 right now')
                    print(f'difference between p2 reference and p1 right now: {abs(p2reference-p2top1)}')
                    print(f'average cosine difference: {avgcosinediff}')   
                    print(f'highest cosine diff: {max(cosinediffs)}')
                    print(f'lowest cosine diff: {min(cosinediffs)}')   
                    """

            # print(f'even though we are working with {otherTrackIds[track_id][0]}, the player id is {playerid}')
            # print(otherTrackIds)
            # If player is already tracked, update their info
            if playerid in players:
                players[playerid].add_pose(kp)
                player_last_positions[playerid] = (x, y)  # Update position
                players[playerid].add_pose(kp)
                # print(f'track id: {track_id}')
                # print(f'playerid: {playerid}')
                if playerid == 1:
                    updated[0][0] = True
                    updated[0][1] = frame_count
                if playerid == 2:
                    updated[1][0] = True
                    updated[1][1] = frame_count
                # print(updated)
                # Player is no longer occluded

                # print(f"Player {playerid} updated.")

            # If the player is new and fewer than MAX_PLAYERS are being tracked
            if len(players) < max_players:
                players[otherTrackIds[track_id][0]] = Player(
                    player_id=otherTrackIds[track_id][1]
                )
                player_last_positions[playerid] = (x, y)
                if playerid == 1:
                    updated[0][0] = True
                    updated[0][1] = frame_count
                else:
                    updated[1][0] = True
                    updated[1][1] = frame_count
                print(f"Player {playerid} added.")

            # putting player keypoints on the frame
            for keypoint in kp:
                # print(keypoint.xyn[0])
                i = 0
                for k in keypoint.xyn[0]:
                    x, y = k
                    x = int(x * frame_width)
                    y = int(y * frame_height)
                    if playerid == 1:
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), 5)
                    else:
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, (255, 0, 0), 5)
                    if i == 16:
                        cv2.putText(
                            annotated_frame,
                            f"{playerid}",
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.5,
                            (255, 255, 255),
                            7,
                        )
                    i += 1
    return [
        pose_model,
        frame,
        otherTrackIds,
        updated,
        references1,
        references2,
        pixdiffs,
        players,
        frame_count,
        player_last_positions,
        frame_width,
        frame_height,
        annotated_frame,
    ]


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


def sum_pixels_in_bbox(frame, bbox):
    x, y, w, h = bbox
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    return np.sum(roi, dtype=np.int64)


import math

from squash import inferenceslicing
def ballplayer_detections(
    frame,
    frame_height,
    frame_width,
    frame_count,
    annotated_frame,
    ballmodel,
    pose_model,
    mainball,
    ball,
    ballmap,
    past_ball_pos,
    ball_false_pos,
    running_frame,
    otherTrackIds,
    updated,
    references1,
    references2,
    pixdiffs,
    players,
    player_last_positions,
):
    ball_detection_results = []
    highestconf = 0
    x1 = x2 = y1 = y2 = 0
    # Ball detection
    ball=ballmodel(frame)

    if Functions.is_ball_false_pos(past_ball_pos, threshold=15):
        ball_false_pos.append(past_ball_pos[-1])
    label = ""
    for box in ball[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = ballmodel.names[int(box.cls)]
        confidence = float(box.conf)  # Convert tensor to float
        int((x1temp + x2temp) / 2)
        int((y1temp + y2temp) / 2)

        if confidence > highestconf:
            highestconf = confidence
            x1 = x1temp
            y1 = y1temp
            x2 = x2temp
            y2 = y2temp

    cv2.rectangle(
        annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
    )

    cv2.putText(
        annotated_frame,
        f"{label} {highestconf:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    # print(label)
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    avg_x = int((x1 + x2) / 2)
    avg_y = int((y1 + y2) / 2)
    size = avg_x * avg_y
    if avg_x > 0 or avg_y > 0:
        if mainball.getlastpos()[0] != avg_x or mainball.getlastpos()[1] != avg_y:
            # print(mainball.getlastpos())
            # print(mainball.getloc())
            mainball.update(avg_x, avg_y, size)
            past_ball_pos.append([avg_x, avg_y, running_frame])
            # print(mainball.getlastpos())
            # print(mainball.getloc())
            math.hypot(
                avg_x - mainball.getlastpos()[0], avg_y - mainball.getlastpos()[1]
            )

            # print(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
            Functions.drawmap(
                mainball.getloc()[0],
                mainball.getloc()[1],
                mainball.getlastpos()[0],
                mainball.getlastpos()[1],
                ballmap,
            )
    """
    FRAMEPOSE
    """
    # going to take frame, sum_pixels_in_bbox, otherTrackIds, updated, player1+2imagereference, pixdiffs, refrences1+2, players,
    framepose_result = framepose(
        pose_model=pose_model,
        frame=frame,
        otherTrackIds=otherTrackIds,
        updated=updated,
        references1=references1,
        references2=references2,
        pixdiffs=pixdiffs,
        players=players,
        frame_count=frame_count,
        player_last_positions=player_last_positions,
        frame_width=frame_width,
        frame_height=frame_height,
        annotated_frame=annotated_frame,
    )
    otherTrackIds = framepose_result[2]
    updated = framepose_result[3]
    references1 = framepose_result[4]
    references2 = framepose_result[5]
    pixdiffs = framepose_result[6]
    players = framepose_result[7]
    player_last_positions = framepose_result[9]
    annotated_frame = framepose_result[12]
    return [
        frame,  # 0
        frame_count,  # 1
        annotated_frame,  # 2
        mainball,  # 3
        ball,  # 4
        ballmap,  # 5
        past_ball_pos,  # 6
        ball_false_pos,  # 7
        running_frame,  # 8
        otherTrackIds,  # 9
        updated,  # 10
        references1,  # 11
        references2,  # 12
        pixdiffs,  # 13
        players,  # 14
        player_last_positions,  # 15
    ]


def slice_frame(width, height, overlap, frame):
    slices = []
    for y in range(0, frame.shape[0], height - overlap):
        for x in range(0, frame.shape[1], width - overlap):
            slice_frame = frame[y : y + height, x : x + width]
            slices.append(slice_frame)
    return slices


def inference_slicing(model, frame, width=100, height=100, overlap=50):
    slices = slice_frame(width, height, overlap, frame)
    results = []
    for slice_frame in slices:
        results.append(model(slice_frame))
    return results
