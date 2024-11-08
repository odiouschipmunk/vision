def main(path="main.mp4"):
    try:
        import cv2
        from ultralytics import YOLO
        import numpy as np
        import math
        from squash import Referencepoints, Functions
        import tensorflow as tf
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from squash.Ball import Ball
        import logging

        from skimage.metrics import structural_similarity as ssim_metric

        print("imported all")

        ball_predict = tf.keras.models.load_model(
            "trained-models/ball_position_model(25k).keras"
        )

        def load_data(file_path):
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
        with open("importantoutput/ball.txt", "w") as f:
            f.write("")
        with open("importantoutput/player1.txt", "w") as f:
            f.write("")
        with open("importantoutput/player2.txt", "w") as f:
            f.write("")
        with open("importantoutput/ball-xyn.txt", "w") as f:
            f.write("")
        with open("importantoutput/read_ball.txt", "w") as f:
            f.write("")
        with open("importantoutput/read_player1.txt", "w") as f:
            f.write("")
        with open("importantoutput/read_player2.txt", "w") as f:
            f.write("")
        pose_model = YOLO("models/yolo11m-pose.pt")
        ballmodel = YOLO("trained-models/g-ball2(white_latest).pt")

        print("loaded models")
        ballvideopath = "output/balltracking.mp4"
        cap = cv2.VideoCapture(path)
        with open("output/final.txt", "w") as f:
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
        future_predict = None
        player_last_positions = {}
        frame_count = 0
        ball_false_pos = []
        past_ball_pos = []
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        output_path = "output/annotated.mp4"
        weboutputpath = "websiteout/annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        importantoutputpath = "output/important.mp4"
        weboutput = cv2.VideoWriter(
            weboutputpath, fourcc, fps, (frame_width, frame_height)
        )
        importantout = cv2.VideoWriter(
            importantoutputpath, fourcc, fps, (frame_width, frame_height)
        )
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        ball_out = cv2.VideoWriter(
            ballvideopath, fourcc, fps, (frame_width, frame_height)
        )
        detections = []

        mainball = Ball(0, 0, 0, 0)
        ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        otherTrackIds = [[0, 0], [1, 1], [2, 2]]
        updated = [[False, 0], [False, 0]]
        reference_points = []
        reference_points = Referencepoints.get_reference_points(
            path=path, frame_width=frame_width, frame_height=frame_height
        )

        references1 = []
        references2 = []

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

        # function to see what kind of shot has been hit
        def shot_type(past_ball_pos, threshold=3):
            # go through the past threshold number of past ball positions and see what kind of shot it is
            # past_ball_pos ordered as [[x,y,frame_number], ...]
            if len(past_ball_pos) < threshold:
                return None
            threshballpos = past_ball_pos[-threshold:]
            # check for crosscourt or straight shots
            xdiff = threshballpos[-1][0] - threshballpos[0][0]
            ydiff = threshballpos[-1][1] - threshballpos[0][1]
            typeofshot = ""
            if xdiff < 50 and ydiff < 50:
                typeofshot = "straight"
            else:
                typeofshot = "crosscourt"
            # check how high the ball has moved
            maxheight = 0
            height = ""
            for i in range(1, len(threshballpos)):
                if threshballpos[i][1] > maxheight:
                    maxheight = threshballpos[i][1]
                    print(f"{threshballpos[i]}")
                    # print(f'maxheight: {maxheight}')
                    # print(f'threshballpos[i][1]: {threshballpos[i][1]}')
            if maxheight < (frame_height) / 1.35:
                height += "lob"
                # print(f'max height was {maxheight} and thresh was {(1.5*frame_height)/2}')
            else:
                height += "drive"
            return typeofshot + " " + height

        def is_match_in_play(
            players,
            mainball,
            movement_threshold=0.2 * frame_width,
            hit=0.15 * frame_height,
        ):
            if players.get(1) is None or players.get(2) is None or mainball is None:
                return False
            try:
                lastplayer1pos = []

                lastplayer2pos = []
                lastballpos = []
                ball_hit = player_move = False
                # lastplayerxpos in the format of [[lanklex, lankley], [ranklex, rankley]]
                lastplayer1pos.append(
                    [
                        players.get(1).get_last_x_poses(1).xyn[0][15][0] * frame_width,
                        players.get(1).get_last_x_poses(1).xyn[0][15][1] * frame_height,
                    ]
                )
                lastplayer2pos.append(
                    [
                        players.get(2).get_last_x_poses(1).xyn[0][15][0] * frame_width,
                        players.get(2).get_last_x_poses(1).xyn[0][15][1] * frame_height,
                    ]
                )
                lastplayer1pos.append(
                    [
                        players.get(1).get_last_x_poses(1).xyn[0][16][0] * frame_width,
                        players.get(1).get_last_x_poses(1).xyn[0][16][1] * frame_height,
                    ]
                )
                lastplayer2pos.append(
                    [
                        players.get(2).get_last_x_poses(1).xyn[0][16][0] * frame_width,
                        players.get(2).get_last_x_poses(1).xyn[0][16][1] * frame_height,
                    ]
                )
                for i in range(1, mainball.number_of_coords()):
                    if mainball.get_last_x_pos(i) is not mainball.get_last_x_pos(i - 1):
                        lastballpos.append(mainball.get_last_x_pos(i))

                # print(f'lastplayer1pos: {lastplayer1pos}')
                lastplayer1distance = math.hypot(
                    lastplayer1pos[0][0] - lastplayer1pos[1][0],
                    lastplayer1pos[0][1] - lastplayer1pos[1][1],
                )
                lastplayer2distance = math.hypot(
                    lastplayer2pos[0][0] - lastplayer2pos[1][0],
                    lastplayer2pos[0][1] - lastplayer2pos[1][1],
                )
                # print(f'lastplayer1distance: {lastplayer1distance}')
                # print(f'lastplayer2distance: {lastplayer2distance}')
                # given that thge ankle position is the 16th and the 17th keypoint, we can check for lunges like so:
                # if the player's ankle moves by more than 5 pixels in the last 5 frames, then the player has lunged
                # if the player has lunged, then the match is in play

                # print(f'last ball pos: {lastballpos}')
                balldistance = math.hypot(
                    lastballpos[0][0] - lastballpos[1][0],
                    lastballpos[0][1] - lastballpos[1][1],
                )
                # print(f'balldistance: {balldistance}')
                if balldistance >= hit:
                    ball_hit = True
                # print(f'last player pos: {lastplayerpos}')
                # print(f'last ball pos: {lastballpos}')
                # print(f'player lunged: {player_move}')
                if (
                    lastplayer1distance >= movement_threshold
                    or lastplayer2distance >= movement_threshold
                ):
                    player_move = True
                # print(f'ball hit: {ball_hit}')
                return [player_move, ball_hit]
            except Exception as e:
                print(
                    f"got exception in is_match_in_play: {e}, line was {e.__traceback__.tb_lineno}"
                )
                return False

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
        homography = Functions.generate_homography(
            reference_points, reference_points_3d
        )
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

        running_frame = 0
        print("started video input")
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        abs(reference_points[1][0] - reference_points[0][0])
        Functions.validate_reference_points(reference_points, reference_points_3d)
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            frame_count += 1

            if running_frame >= 500:
                pass
            if frame_count >= 9000:
                cap.release()
                cv2.destroyAllWindows()
            if len(references1) != 0 and len(references2) != 0:
                sum(references1) / len(references1)
                sum(references2) / len(references2)

            running_frame += 1

            if running_frame == 1:
                print("frame 1")
                courtref = np.int64(
                    Functions.sum_pixels_in_bbox(
                        frame, [0, 0, frame_width, frame_height]
                    )
                )
                print(courtref)
                referenceimage = frame

            if is_camera_angle_switched(frame, referenceimage, threshold=0.5):
                print("camera angle switched")
                continue

            currentref = int(
                Functions.sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
            )

            if abs(courtref - currentref) > courtref * 0.6:
                print("most likely not original camera frame")
                print("current ref: ", currentref)
                print("court ref: ", courtref)
                print(f"frame count: {frame_count}")
                print(
                    f"difference between current ref and court ref: {abs(courtref - currentref)}"
                )
                continue

            ball = ballmodel(frame)
            detections.append(ball)

            annotated_frame = frame.copy()  # pose_results[0].plot()

            for reference in reference_points:
                cv2.circle(
                    annotated_frame,
                    (int(reference[0]), int(reference[1])),
                    5,
                    (0, 255, 0),
                    2,
                )

            # frame, frame_height, frame_width, frame_count, annotated_frame, ballmodel, pose_model, mainball, ball, ballmap, past_ball_pos, ball_false_pos, running_frame
            # framepose_result=framepose.framepose(pose_model=pose_model, frame=frame, otherTrackIds=otherTrackIds, updated=updated, references1=references1, references2=references2, pixdiffs=pixdiffs, players=players, frame_count=frame_count, player_last_positions=player_last_positions, frame_width=frame_width, frame_height=frame_height, annotated_frame=annotated_frame)
            detections_result = Functions.ballplayer_detections(
                frame=frame,
                frame_height=frame_height,
                frame_width=frame_width,
                frame_count=frame_count,
                annotated_frame=annotated_frame,
                ballmodel=ballmodel,
                pose_model=pose_model,
                mainball=mainball,
                ball=ball,
                ballmap=ballmap,
                past_ball_pos=past_ball_pos,
                ball_false_pos=ball_false_pos,
                running_frame=running_frame,
                otherTrackIds=otherTrackIds,
                updated=updated,
                references1=references1,
                references2=references2,
                pixdiffs=pixdiffs,
                players=players,
                player_last_positions=player_last_positions,
            )
            frame = detections_result[0]
            frame_count = detections_result[1]
            annotated_frame = detections_result[2]
            mainball = detections_result[3]
            ball = detections_result[4]
            ballmap = detections_result[5]
            past_ball_pos = detections_result[6]
            ball_false_pos = detections_result[7]
            running_frame = detections_result[8]
            otherTrackIds = detections_result[9]
            updated = detections_result[10]
            references1 = detections_result[11]
            references2 = detections_result[12]
            pixdiffs = detections_result[13]
            players = detections_result[14]
            player_last_positions = detections_result[15]

            # print(f'is match in play: {is_match_in_play(players, mainball)}')
            match_in_play = is_match_in_play(players, mainball)
            type_of_shot = Functions.classify_shot(
                past_ball_pos, homography_matrix=homography
            )
            if match_in_play is not False:
                # print(match_in_play)
                cv2.putText(
                    annotated_frame,
                    f"ball hit: {str(match_in_play[1])}",
                    (10, frame_height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    f"player legs far apart: {str(match_in_play[0])}",
                    (10, frame_height - 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    f"shot type: {type_of_shot}",
                    (10, frame_height - 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
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
                            players.get(1).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p1_right_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p1_right_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][15][1]
                            * frame_height
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
                            players.get(2).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p2_right_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p2_right_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][15][1]
                            * frame_height
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

            # Generate player ankle heatmap
            if (
                players.get(1).get_latest_pose() is not None
                and players.get(2).get_latest_pose() is not None
            ):
                player_ankles = [
                    (
                        int(
                            players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
                        ),
                        int(
                            players.get(1).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        ),
                    ),
                    (
                        int(
                            players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
                        ),
                        int(
                            players.get(2).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        ),
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
                cv2.circle(
                    annotated_frame,
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    7,
                    (0, 0, 255),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 1 frame: {int(predicted_pos[0][0]*frame_width)},{int(predicted_pos[0][1]*frame_height)}",
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                last9 = positions[-9:]
                last9.append([predicted_pos[0][0], predicted_pos[0][1]])
                # print(f'last 9: {last9}')
                sequence_and_predicted = np.array(last9)
                # print(f'sequence and predicted: {sequence_and_predicted}')
                sequence_and_predicted = sequence_and_predicted.reshape((1, 10, 2, 1))
                future_predict = ball_predict(sequence_and_predicted)
                cv2.circle(
                    annotated_frame,
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    7,
                    (255, 0, 0),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 3 frames: {int(future_predict[0][0]*frame_width)},{int(future_predict[0][1]*frame_height)}",
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
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
                with open("output/final.txt", "a") as f:
                    text = f"Frame: {running_frame}{{Player 1: {p1postemp}\nPlayer 2: {p2postemp}\nBall: {mainball.getloc()}\nType of shot: {type_of_shot[0]+type_of_shot[1]}\nBall hit: {str(match_in_play[1])}\nWalls hit: {type_of_shot[2]}}}\n"
                    f.write(f"{text}\n")
                    f.close()

            def importantwrite():
                with open("importantoutput/read_player1.txt", "a") as f:
                    f.write(f"{p1postemp}\n")
                    f.close()
                with open("importantoutput/read_player2.txt", "a") as f:
                    f.write(f"{p2postemp}\n")
                    f.close()
                with open("importantoutput/player1.txt", "a") as f:
                    for pos in p1postemp:
                        f.write(f"{pos[0]}\n{pos[1]}\n")
                    f.close()
                with open("importantoutput/player2.txt", "a") as f:
                    for pos in p2postemp:
                        f.write(f"{pos[0]}\n{pos[1]}\n")
                    f.close()
                with open("importantoutput/ball.txt", "a") as f:
                    f.write(f"{mainball.getloc()[0]}\n{mainball.getloc()[1]}\n")
                with open("importantoutput/read_ball.txt", "a") as f:
                    f.write(f"{mainball.getloc()}\n")
                with open("importantoutput/ball-xyn.txt", "a") as f:
                    f.write(
                        f"{mainball.getloc()[0]/frame_width}\n{mainball.getloc()[1]/frame_height}\n"
                    )

            try:
                if running_frame % 3 == 0:
                    write()
            except Exception as e:
                print(
                    f"could not write to file, most likely because players were not detected yet: {e}"
                )
                # print(f'line was {e.__traceback__.tb_lineno}')
                # print(f'all info about e: ')
            try:
                # print(match_in_play)

                if match_in_play[0] or match_in_play[1]:
                    # print(f'wrote to important!')
                    importantout.write(annotated_frame)
                    importantwrite()
            except Exception as e:
                print(f"probably not enough info: {e}")
            ball_out.write(annotated_frame)
            out.write(annotated_frame)
            weboutput.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"error2: {e}")
        print(f'line was {e.__traceback__.tb_lineno}')
        print(f'other into about e: {e.__traceback__}')
        print(f'other info about e: {e.__traceback__.tb_frame}')
        print(f'other info about e: {e.__traceback__.tb_next}')
        print(f'other info about e: {e.__traceback__.tb_lasti}')
        
        


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"error: {e}")
