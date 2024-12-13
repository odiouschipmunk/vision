def main():
    import csv
    import ast
    frames=[]
    with open('output/final.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            frames.append(row)
    print(ast.literal_eval(frames[1][1])[0])
if __name__ == "__main__":
    main()

'''
def main():
    """
    The `main` function processes video frames to detect players, their poses, and the ball trajectory
    in a squash game.
    """
    import cv2
    from ultralytics import YOLO
    import numpy as np
    import math
    from squash import Referencepoints, Functions
    import tensorflow as tf
    import matplotlib
    import framepose

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from squash.Ball import Ball
    import logging

    # from squash.Player import Player
    # from PIL import Image
    from skimage.metrics import structural_similarity as ssim_metric

    print("imported all")
    # Define the reference points in pixel coordinates (image)
    # These should be the coordinates of the reference points in the image
    # TODO: use embeddings to correctly find the different players
    ball_predict = tf.keras.models.load_model(
        "trained-models/ball_position_model(25k).keras"
    )

    def load_data(file_path):
        """
        Load ball positions from the file and return a list of 2D tuples.
        """
        with open(file_path, "r") as file:
            data = file.readlines()

        # Convert the data to a list of floats
        data = [float(line.strip()) for line in data]

        # Group the data into pairs of coordinates (x, y)
        positions = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]

        return positions

    with open("output/ball.txt", "w") as f:
        f.write("")
    with open("output/player1.txt", "w") as f:
        f.write("")
    with open("output/player2.txt", "w") as f:
        f.write("")
    with open("output/ball-xyn.txt", "w") as f:
        f.write("")
    with open("output/read_ball.txt", "w") as f:
        f.write("")
    with open("output/read_player1.txt", "w") as f:
        f.write("")
    with open("output/read_player2.txt", "w") as f:
        f.write("")
    # Load models
    pose_model = YOLO("models/yolo11m-pose.pt")
    ballmodel = YOLO("trained-models/g-ball2(white_latest).pt")
    # racketmodel=YOLO('trained-models/squash-racket.pt')
    # courtmodel=YOLO('trained-models/court-key!.pt')
    # Video file path
    path = "main.mp4"
    print("loaded models")
    ballvideopath = "output/balltracking.mp4"
    cap = cv2.VideoCapture(path)
    with open("output/final.txt", "a") as f:
        f.write(
            f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n"
        )
    frame_width = 640
    frame_height = 360
    players = {}
    courtref = 0
    occlusion_times = {}
    for i in range(1, 3):
        occlusion_times[i] = 0
    # Get video dimensions
    player_last_positions = {}
    frame_count = 0
    ball_false_pos = []
    past_ball_pos = []
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    output_path = "output/annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file
    fps = 25  # Frames per second
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    ball_out = cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))
    detections = []
    # in the form of ball, [pose], track
    # Create a blank canvas for heatmap based on video resolution

    mainball = Ball(0, 0, 0, 0)
    ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    # other track ids necessary as since players get occluded, im just going to assign that track id to the previous id(1 or 2) to the last occluded player
    # really need to fix this as if there are 2 occluded players, it will not work
    otherTrackIds = [[0, 0], [1, 1], [2, 2]]
    updated = [[False, 0], [False, 0]]
    reference_points = []
    reference_points = Referencepoints.get_reference_points(
        path=path, frame_width=frame_width, frame_height=frame_height
    )

    references1 = []
    references2 = []
    """
    def findRef(img):
        return cv2.
    """

    pixdiffs = []


    p1distancesfromT = []
    p2distancesfromT = []

    courtref = np.int64(courtref)
    referenceimage = None

    def is_camera_angle_switched(frame, reference_image, threshold=0.5):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        score, _ = ssim_metric(reference_image_gray, frame_gray, full=True)
        return score < threshold

    # note for anyone else seeing this:
    # reference[0] is x val and [1] is y val
    # reference[0] is top left,
    # reference[1] is top right
    # reference[2] is bottom right
    # reference[3] is bottom left [0,0,0]
    # reference[4] is T
    # reference[5] is left bottom of service box
    # reference[6] is right bottom of service box
    # references[7] is left of tin
    # references[8] is right of tin
    # reference[9] is left of service line
    # reference[10] is right of service line
    # reference[11] is left of the top line of the front court
    # reference[12] is right of the top line of the front court
    reference_points_3d = [
        [0, 0, 9.75],  # Top-left corner, 1
        [6.4, 0, 9.75],  # Top-right corner, 2
        [6.4, 0, 0],  # Bottom-right corner, 3
        [0, 0, 0],  # Bottom-left corner, 4
        [3.2, 0, 4.26],  # "T" point, 5
        [0, 0, 2.66],  # Left bottom of the service box, 6
        [6.4, 0, 2.66],  # Right bottom of the service box, 7
        [0, 0.48, 9.75],  # left of tin, 8
        [6.4, 0.48, 9.75],  # right of tin, 9
        [0, 1.83, 9.75],  # Left of the service line, 10
        [4.8, 1.83, 9.75],  # Right of the service line, 11
        [0, 4.57, 9.75],  # Left of the top line of the front court, 12
        [6.4, 4.57, 9.75],  # Right of the top line of the front court, 13
    ]
    homography = Functions.generate_homography(reference_points, reference_points_3d)
    np.zeros((frame_height, frame_width), dtype=np.float32)
    np.zeros((frame_height, frame_width), dtype=np.float32)
    heatmap_overlay_path = "output/white.png"
    heatmap_image = cv2.imread(heatmap_overlay_path)
    if heatmap_image is None:
        raise FileNotFoundError(
            f"Could not find heatmap overlay image at {heatmap_overlay_path}"
        )
    np.zeros_like(heatmap_image, dtype=np.float32)

    ballxy = []

    # homography=Functions.calculate_homography(reference_points, reference_points_3d)

    running_frame = 0
    print("started video input")
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Functions.validate_reference_points(reference_points, reference_points_3d)
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # make it so that annotated frame goes from left to right and then goes back to left and repeats again
        # annotated frame should be the size of sliced_x and sliced_y and shouldn't be just resizing the frame to that size
        # instead, it should be taking the frame and then slicing it into the size of sliced_x and sliced_y
        # then, it should be taking the sliced frame and then putting it into the annotated frame
        # after that, it should be going left to right
        # if it reaches the end of the frame, then it should go back to the left and then go down
        frame_count += 1
        # x_steps = range(0, frame_width, slice_width - overlap)
        # y_steps = range(0, frame_height, slice_height - overlap)
        # detections=[]
        # for y in y_steps:
        #     for x in x_steps:
        #         x_start=x
        #         y_start=y
        #         x_end=min(x_start+slice_width, frame_width)
        #         y_end=min(y_start+slice_height, frame_height)
        #         slice_frame=frame[y_start:y_end, x_start:x_end]
        if running_frame >= 500:
            pass
        if frame_count >= 25000:
            cap.release()
            cv2.destroyAllWindows()
        if len(references1) != 0 and len(references2) != 0:
            sum(references1) / len(references1)
            sum(references2) / len(references2)
        running_frame += 1
        frame = cv2.resize(frame, (frame_width, frame_height))
        """
        if len(p1embeddings) != 0 and len(p2embeddings) != 0 and len(p1embeddings) > 1 and len(p2embeddings) > 1:
            #print(len(p1embeddings[-1]))
            #print(p1embeddings)
            similarity_p1 = cosine_similarity(p1embeddings[-1], p2embeddings[-1])
            similarity_p2 = cosine_similarity(p2embeddings[-1], p1embeddings[-1])
            print(f"Cosine Similarity p1: {similarity_p1}")
            print(f"Cosine Similarity p2: {similarity_p2}")
        """
        # if (biggestx == 0 or biggesty == 0 or smallestx == 0 or smallesty == 0) and len(reference_points) == 11:
        #     biggestx=reference_points[2][0]
        #     biggesty=reference_points[9][1]
        #     smallestx=reference_points[0][0]
        #     smallesty=reference_points[3][1]
        #     frame=frame[smallesty:biggesty, smallestx:biggestx]

        if running_frame == 1:
            print("frame 1")
            courtref = np.int64(
                Functions.sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
            )
            print(courtref)
            referenceimage = frame

        if is_camera_angle_switched(frame, referenceimage, threshold=0.5):
            print("camera angle switched")
            continue

        # print(len(players))

        currentref = int(
            Functions.sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
        )

        # general court reference to only get the first camera angle throughout the video
        if abs(courtref - currentref) > courtref * 0.6:
            print("most likely not original camera frame")
            print("current ref: ", currentref)
            print("court ref: ", courtref)
            print(f"frame count: {frame_count}")
            print(
                f"difference between current ref and court ref: {abs(courtref - currentref)}"
            )
            continue
        # if biggestx == 0 or biggesty == 0 or smallestx == 0 or smallesty == 0:
        #     original_width, original_height = frame.shape[:2]

        #     # Set largest and smallest x and y coordinates based on `reference_points`
        #     biggestx = max(reference_points[2][0], reference_points[1][0])
        #     biggesty = max(reference_points[3][1], reference_points[2][1])
        #     smallestx = min(reference_points[0][0], reference_points[3][0])
        #     smallesty = min(reference_points[9][1], reference_points[10][1])

        #     if smallestx < biggestx and smallesty < biggesty:
        #         try:
        #             cropped_frame = frame[smallesty:biggesty, smallestx:biggestx]
        #             frame = cv2.resize(cropped_frame, (original_width, original_height))
        #         except cv2.error:
        #             print("Error during crop/resize operation - using original frame")
        #     else:
        #         print("Invalid crop dimensions - using original frame")

        #     # Debug print
        #     print(f"biggest x: {biggestx}, biggest y: {biggesty}, smallest x: {smallestx}, smallest y: {smallesty}")

        # Pose and ball detection
        ball = ballmodel(frame)
        detections.append(ball)
        # pose_results = pose_model(frame)
        # racket_results=racketmodel(frame)
        # only plot the top 2 confs
        annotated_frame = frame.copy()  # pose_results[0].plot()

        # court_results=courtmodel(frame)
        # Check if keypoints exist and are not empty
        # print(pose_results)

        # false_pos=Functions.ball_is_false_positive(past_ball_pos)
        # if false_pos is not None:
        # ball_false_pos.append(false_pos)
        # print(f'ball false pos: {ball_false_pos}')
        highestconf = 0
        x1 = x2 = y1 = y2 = 0
        # Ball detection
        # make it so that if it detects the ball in the same place multiple times it takes that out
        if Functions.is_ball_false_pos(past_ball_pos, threshold=15):
            ball_false_pos.append(past_ball_pos[-1])
            # print(f'ball false pos: {ball_false_pos}, with last 5 past ball pos as : {past_ball_pos[-5:]}')
        label = ""
        for box in ball[0].boxes:
            coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
            x1temp, y1temp, x2temp, y2temp = coords
            label = ballmodel.names[int(box.cls)]
            confidence = float(box.conf)  # Convert tensor to float
            int((x1temp + x2temp) / 2)
            int((y1temp + y2temp) / 2)
            # if len(ball_false_pos) > 0:
            #     if avgxtemp in ball_false_pos[0] and avgytemp in ball_false_pos[1]:
            #         continue
            """
            if abs(avgxtemp-363)<10 and abs(avgytemp-72)<10:
                #false positive near the "V"
                #TODO find out how to check for false positives for general videos
                continue
            """
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
        framepose_result = framepose.framepose(
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

        for reference in reference_points:
            cv2.circle(
                annotated_frame,
                (int(reference[0]), int(reference[1])),
                5,
                (0, 255, 0),
                2,
            )

        # Save the heatmap
        # print(players)
        # print(players.get(1).get_latest_pose())
        # print(players.get(2).get_latest_pose())

        # print(len(players))

        # Display ankle positions of both players
        if players.get(1) and players.get(2) is not None:
            # print('line 263')
            # print(f'players: {players}')
            # print(f'players 1: {players.get(1)}')
            # print(f'players 2: {players.get(2)}')
            # print(f'players 1 latest pose: {players.get(1).get_latest_pose()}')
            # print(f'players 2 latest pose: {players.get(2).get_latest_pose()}')
            if (
                players.get(1).get_latest_pose()
                or players.get(2).get_latest_pose() is not None
            ):
                # print('line 265')
                try:
                    p1_left_ankle_x = int(
                        players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
                    )
                    p1_left_ankle_y = int(
                        players.get(1).get_latest_pose().xyn[0][16][1] * frame_height
                    )
                    p1_right_ankle_x = int(
                        players.get(1).get_latest_pose().xyn[0][15][0] * frame_width
                    )
                    p1_right_ankle_y = int(
                        players.get(1).get_latest_pose().xyn[0][15][1] * frame_height
                    )
                except Exception:
                    p1_left_ankle_x = p1_left_ankle_y = p1_right_ankle_x = (
                        p1_right_ankle_y
                    ) = 0
                try:
                    p2_left_ankle_x = int(
                        players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
                    )
                    p2_left_ankle_y = int(
                        players.get(2).get_latest_pose().xyn[0][16][1] * frame_height
                    )
                    p2_right_ankle_x = int(
                        players.get(2).get_latest_pose().xyn[0][15][0] * frame_width
                    )
                    p2_right_ankle_y = int(
                        players.get(2).get_latest_pose().xyn[0][15][1] * frame_height
                    )
                except Exception:
                    p2_left_ankle_x = p2_left_ankle_y = p2_right_ankle_x = (
                        p2_right_ankle_y
                    ) = 0
                # Display the ankle positions on the bottom left of the frame
                avgxank1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                avgyank1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                avgxank2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                avgyank2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                text_p1 = f"P1 position(ankle): {avgxank1},{avgyank1}"
                cv2.putText(
                    annotated_frame,
                    f"{otherTrackIds[Functions.findLast(1, otherTrackIds)][1]}",
                    (p1_left_ankle_x, p1_left_ankle_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    f"{otherTrackIds[Functions.findLast(2, otherTrackIds)][1]}",
                    (p2_left_ankle_x, p2_left_ankle_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                text_p2 = f"P2 position(ankle): {avgxank2},{avgyank2}"
                cv2.putText(
                    annotated_frame,
                    text_p1,
                    (10, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    text_p2,
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                avgpx1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                avgpy1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                avgpx2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                avgpy2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                # print(reference_points)
                p1distancefromT = math.hypot(
                    reference_points[4][0] - avgpx1, reference_points[4][1] - avgpy1
                )
                p2distancefromT = math.hypot(
                    reference_points[4][0] - avgpx2, reference_points[4][1] - avgpy2
                )
                p1distancesfromT.append(p1distancefromT)
                p2distancesfromT.append(p2distancefromT)
                text_p1t = f"P1 distance from T: {p1distancesfromT[-1]}"
                text_p2t = f"P2 distance from T: {p2distancesfromT[-1]}"
                cv2.putText(
                    annotated_frame,
                    text_p1t,
                    (10, frame_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    text_p2t,
                    (10, frame_height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                plt.figure(figsize=(10, 6))
                plt.plot(p1distancesfromT, color="blue", label="P1 Distance from T")
                plt.plot(p2distancesfromT, color="red", label="P2 Distance from T")

                # Add labels and title
                plt.xlabel("Time (frames)")
                plt.ylabel("Distance from T")
                plt.title("Distance from T over Time")
                plt.legend()

                # Save the plot to a file
                plt.savefig("output/distance_from_t_over_time.png")

                # Close the plot to free up memory
                plt.close()

        # Display the annotated frame
        """
        COURT DETECTION
        for box in court[0].boxes:
            coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
            x1temp, y1temp, x2temp, y2temp = coords
            label = courtmodel.names[int(box.cls)]
            confidence = float(box.conf)
            cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    """

        # Generate player ankle heatmap
        if (
            players.get(1).get_latest_pose() is not None
            and players.get(2).get_latest_pose() is not None
        ):
            player_ankles = [
                (
                    int(players.get(1).get_latest_pose().xyn[0][16][0] * frame_width),
                    int(players.get(1).get_latest_pose().xyn[0][16][1] * frame_height),
                ),
                (
                    int(players.get(2).get_latest_pose().xyn[0][16][0] * frame_width),
                    int(players.get(2).get_latest_pose().xyn[0][16][1] * frame_height),
                ),
            ]

            # Draw points on the heatmap
            for ankle in player_ankles:
                cv2.circle(
                    heatmap_image, ankle, 5, (255, 0, 0), -1
                )  # Blue points for Player 1
                cv2.circle(
                    heatmap_image, ankle, 5, (0, 0, 255), -1
                )  # Red points for Player 2

        blurred_heatmap_ankle = cv2.GaussianBlur(heatmap_image, (51, 51), 0)

        # Normalize heatmap and apply color map in one step
        normalized_heatmap = cv2.normalize(
            blurred_heatmap_ankle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        heatmap_overlay = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

        # Combine with white image
        cv2.addWeighted(
            np.ones_like(heatmap_overlay) * 255, 0.5, heatmap_overlay, 0.5, 0
        )

        # Save the combined image
        # cv2.imwrite("output/heatmap_ankle.png", combined_image)
        ballx = bally = 0
        # ball stuff
        if (
            mainball is not None
            and mainball.getlastpos() is not None
            and mainball.getlastpos() != (0, 0)
        ):
            ballx = mainball.getlastpos()[0]
            bally = mainball.getlastpos()[1]
            if ballx != 0 and bally != 0:
                if [ballx, bally] not in ballxy:
                    ballxy.append([ballx, bally, frame_count])
                    # print(
                    #     f"ballx: {ballx}, bally: {bally}, appended to ballxy with length {len(ballxy)} and frame count as : {frame_count}"
                    # )

        # Draw the ball trajectory
        if len(ballxy) > 2:
            for i in range(1, len(ballxy)):
                if ballxy[i - 1] is None or ballxy[i] is None:
                    continue
                if ballxy[i][2] - ballxy[i - 1][2] < 7:
                    if frame_count - ballxy[i][2] < 7:
                        cv2.line(
                            annotated_frame,
                            (ballxy[i - 1][0], ballxy[i - 1][1]),
                            (ballxy[i][0], ballxy[i][1]),
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(
                            annotated_frame,
                            (ballxy[i - 1][0], ballxy[i - 1][1]),
                            5,
                            (0, 255, 0),
                            -1,
                        )
                        cv2.circle(
                            annotated_frame,
                            (ballxy[i][0], ballxy[i][1]),
                            5,
                            (0, 255, 0),
                            -1,
                        )

                        # cv2.circle(
                        #    annotated_frame,
                        #    (next_pos[0], next_pos[1]),
                        #    5,
                        #    (0, 255, 0),
                        #    -1,
                        # )

        for ball_pos in ballxy:
            if frame_count - ball_pos[2] < 7:
                # print(f'wrote to frame on line 1028 with coords: {ball_pos}')
                cv2.circle(
                    annotated_frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 0), -1
                )

        positions = load_data("output\\ball-xyn.txt")
        if len(positions) > 11:
            input_sequence = np.array([positions[-10:]])
            input_sequence = input_sequence.reshape((1, 10, 2, 1))
            predicted_pos = ball_predict(input_sequence)
            # print(f'input_sequence: {input_sequence}')
            # cv2.circle(
            #     annotated_frame,
            #     (
            #         int(predicted_pos[0][0] * frame_width),
            #         int(predicted_pos[0][1] * frame_height),
            #     ),
            #     7,
            #     (0, 0, 255),
            #     7,
            # )
            # cv2.putText(
            #     annotated_frame,
            #     f"predicted ball position in 1 frame: {int(predicted_pos[0][0]*frame_width)},{int(predicted_pos[0][1]*frame_height)}",
            #     (
            #         int(predicted_pos[0][0] * frame_width),
            #         int(predicted_pos[0][1] * frame_height),
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
            last9 = positions[-9:]
            last9.append([predicted_pos[0][0], predicted_pos[0][1]])
            # print(f'last 9: {last9}')
            sequence_and_predicted = np.array(last9)
            # print(f'sequence and predicted: {sequence_and_predicted}')
            sequence_and_predicted = sequence_and_predicted.reshape((1, 10, 2, 1))
            # future_predict = ball_predict(sequence_and_predicted)
            # cv2.circle(
            #     annotated_frame,
            #     (
            #         int(future_predict[0][0] * frame_width),
            #         int(future_predict[0][1] * frame_height),
            #     ),
            #     7,
            #     (255, 0, 0),
            #     7,
            # )
            # cv2.putText(
            #     annotated_frame,
            #     f"predicted ball position in 3 frames: {int(future_predict[0][0]*frame_width)},{int(future_predict[0][1]*frame_height)}",
            #     (
            #         int(future_predict[0][0] * frame_width),
            #         int(future_predict[0][1] * frame_height),
            #     ),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
        if (
            players.get(1)
            and players.get(2) is not None
            and (
                players.get(1).get_last_x_poses(3) is not None
                and players.get(2).get_last_x_poses(3) is not None
            )
        ):
            p1postemp = players.get(1).get_last_x_poses(3).xyn[0]
            p2postemp = players.get(2).get_last_x_poses(3).xyn[0]
            rlp1postemp = [
                players.get(1).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                players.get(1).get_last_x_poses(3).xyn[0][16][1] * frame_height,
            ]
            rlp2postemp = [
                players.get(2).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                players.get(2).get_last_x_poses(3).xyn[0][16][1] * frame_height,
            ]
            rlworldp1 = Functions.pixel_to_3d(
                rlp1postemp, homography, reference_points_3d
            )
            rlworldp2 = Functions.pixel_to_3d(
                rlp2postemp, homography, reference_points_3d
            )
            text5 = f"Player 1: {rlworldp1}"
            text6 = f"Player 2: {rlworldp2}"

            cv2.putText(
                annotated_frame,
                text5,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                annotated_frame,
                text6,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            # cv2.putText(
            #     annotated_frame,
            #     text7,
            #     (10, 90),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
            # cv2.putText(
            #     annotated_frame,
            #     text8,
            #     (10, 110),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
            # Functions.transform_and_display(rlp1postemp, rlp2postemp, pixel_reference=reference_points, reference_points_3d=reference_points_3d, image=annotated_frame)
        if len(ballxy) > 0:
            balltext = f"Ball position: {ballxy[-1][0]},{ballxy[-1][1]}"
            rlball = Functions.pixel_to_3d(
                [ballxy[-1][0], ballxy[-1][1]], homography, reference_points_3d
            )
            text4 = f"Ball position in world: {rlball}"
            cv2.putText(
                annotated_frame,
                balltext,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                annotated_frame,
                text4,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        def write():
            with open("output/read_player1.txt", "a") as f:
                f.write(f"{p1postemp}\n")
                f.close()
            with open("output/read_player2.txt", "a") as f:
                f.write(f"{p2postemp}\n")
                f.close()
            with open("output/player1.txt", "a") as f:
                for pos in p1postemp:
                    f.write(f"{pos[0]}\n{pos[1]}\n")
                f.close()
            with open("output/player2.txt", "a") as f:
                for pos in p2postemp:
                    f.write(f"{pos[0]}\n{pos[1]}\n")
                f.close()
            with open("output/ball.txt", "a") as f:
                f.write(f"{mainball.getloc()[0]}\n{mainball.getloc()[1]}\n")
            with open("output/read_ball.txt", "a") as f:
                f.write(f"{mainball.getloc()}\n")
            with open("output/ball-xyn.txt", "a") as f:
                f.write(
                    f"{mainball.getloc()[0]/frame_width}\n{mainball.getloc()[1]/frame_height}\n"
                )
            """
            with open("output/final.txt", "a") as f:
                text = f"Frame: {running_frame}{{\nPlayer 1: {p1postemp}\n\nPlayer 2: {p2postemp}\n\nBall: {mainball.getloc()}}}\n\n"
                f.write(f"{text}\n")
                f.close()
            """
            # print(f'wrote!')

        if running_frame % 3 == 0:
            try:
                write()
            except Exception as e:
                print(
                    f"could not write to file, most likely because players were not detected yet: {e}"
                )

        # most_likely_ballframe=[int(future_predict[0][1]*frame_width), int(future_predict[0][1]*frame_height)]
        # ball_frame=frame[most_likely_ballframe[0]-50:most_likely_ballframe[0]+50, most_likely_ballframe[1]-50:most_likely_ballframe[1]+50]
        # im aweseome
        ball_out.write(annotated_frame)
        out.write(annotated_frame)
        cv2.imshow("Annotated Frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''