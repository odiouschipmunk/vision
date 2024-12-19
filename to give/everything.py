import cv2
import math
import time
import csv
import csvanalyze
import matplotlib
import json
import os
import clip
import torch
import numpy as np
import tensorflow as tf
import logging
from ultralytics import YOLO
from matplotlib import pyplot as plt
from squash.Ball import Ball
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from transformers import AutoModelForCausalLM, AutoTokenizer
from skimage.metrics import structural_similarity as ssim_metric
from norfair import Tracker
from squash.Player import Player
from norfair.filter import OptimizedKalmanFilterFactory

matplotlib.use("Agg")

start = time.time()
alldata = organizeddata = []


def main(path="main_laptop.mp4", frame_width=640, frame_height=360):
    try:
        print("imported all")
        csvstart = 0
        end = csvstart + 100
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

        cleanwrite()
        pose_model = YOLO("models/yolo11n-pose.pt")
        ballmodel = YOLO("trained-models\\g-ball2(white_latest).pt")

        print("loaded models")
        ballvideopath = "output/balltracking.mp4"
        cap = cv2.VideoCapture(path)
        with open("output/final.txt", "w") as f:
            f.write(
                f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n"
            )

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
        cv2.VideoWriter(weboutputpath, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(importantoutputpath, fourcc, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))
        detections = []

        mainball = Ball(0, 0, 0, 0)
        ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        otherTrackIds = [[0, 0], [1, 1], [2, 2]]
        updated = [[False, 0], [False, 0]]
        reference_points = []
        reference_points = get_reference_points(
            path=path, frame_width=frame_width, frame_height=frame_height
        )

        references1 = []
        references2 = []

        pixdiffs = []

        p1distancesfromT = []
        p2distancesfromT = []

        courtref = np.int64(courtref)
        referenceimage = None

        # function to see what kind of shot has been hit

        reference_points_3d = [
            [0, 9.75, 0],  # Top-left corner, 1
            [6.4, 9.75, 0],  # Top-right corner, 2
            [6.4, 0, 0],  # Bottom-right corner, 3
            [0, 0, 0],  # Bottom-left corner, 4
            [3.2, 0, 4.31],  # "T" point, 5
            [0, 2.71, 0],  # Left bottom of the service box, 6
            [6.4, 2.71, 0],  # Right bottom of the service box, 7
            [0, 9.75, 0.48],  # left of tin, 8
            [6.4, 9.75, 0.48],  # right of tin, 9
            [0, 9.75, 1.78],  # Left of the service line, 10
            [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
            [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
        ]
        homography = generate_homography(reference_points, reference_points_3d)

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
        validate_reference_points(reference_points, reference_points_3d)
        print(f"loaded everything in {time.time()-start} seconds")
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            # force it to go to lowestx-->highestx and then lowesty-->highesty
            frame_count += 1

            if len(references1) != 0 and len(references2) != 0:
                sum(references1) / len(references1)
                sum(references2) / len(references2)

            running_frame += 1
            if running_frame == 1:
                courtref = np.int64(
                    sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
                )
                referenceimage = frame

            if is_camera_angle_switched(frame, referenceimage, threshold=0.5):
                continue

            currentref = int(
                sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
            )

            if abs(courtref - currentref) > courtref * 0.6:
                # print("most likely not original camera frame")
                # print("current ref: ", currentref)
                # print("court ref: ", courtref)
                # print(f"frame count: {frame_count}")
                # print(
                #     f"difference between current ref and court ref: {abs(courtref - currentref)}"
                # )
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
            print("loaded models")
            # frame, frame_height, frame_width, frame_count, annotated_frame, ballmodel, pose_model, mainball, ball, ballmap, past_ball_pos, ball_false_pos, running_frame
            # framepose_result=framepose.framepose(pose_model=pose_model, frame=frame, otherTrackIds=otherTrackIds, updated=updated, references1=references1, references2=references2, pixdiffs=pixdiffs, players=players, frame_count=frame_count, player_last_positions=player_last_positions, frame_width=frame_width, frame_height=frame_height, annotated_frame=annotated_frame)
            detections_result = ballplayer_detections(
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
                other_track_ids=otherTrackIds,
                updated=updated,
                references1=references1,
                references2=references2,
                pixdiffs=pixdiffs,
                players=players,
                player_last_positions=player_last_positions,
                occluded=False,
                importantdata=[],
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
            detections_result[16]
            idata = detections_result[17]
            if idata:
                alldata.append(idata)
            # print(f"occluded: {occluded}")
            # occluded structured as [[players_found, last_pos_p1, last_pos_p2, frame_number]...]
            # print(f'is match in play: {is_match_in_play(players, mainball)}')
            match_in_play = is_match_in_play(players, mainball)
            type_of_shot = classify_shot(past_ball_pos=past_ball_pos)
            try:
                reorganize_shots(alldata)
                # print(f"organized data: {organizeddata}")
            except Exception as e:
                print(f"error reorganizing: {e}")
                pass
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
                        f"{otherTrackIds[find_last(1, otherTrackIds)][1]}",
                        (p1_left_ankle_x, p1_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[find_last(2, otherTrackIds)][1]}",
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
            try:
                # Generate player ankle heatmap
                if (
                    players.get(1).get_latest_pose() is not None
                    and players.get(2).get_latest_pose() is not None
                ):
                    player_ankles = [
                        (
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][0]
                                * frame_width
                            ),
                            int(
                                players.get(1).get_latest_pose().xyn[0][16][1]
                                * frame_height
                            ),
                        ),
                        (
                            int(
                                players.get(2).get_latest_pose().xyn[0][16][0]
                                * frame_width
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
                heatmap_overlay = cv2.applyColorMap(
                    normalized_heatmap, cv2.COLORMAP_JET
                )

                # Combine with white image
                cv2.addWeighted(
                    np.ones_like(heatmap_overlay) * 255, 0.5, heatmap_overlay, 0.5, 0
                )
            except Exception:
                # print(f"line618: {e}")
                pass
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
                players.get(1).get_last_x_poses(3).xyn[0]
                players.get(2).get_last_x_poses(3).xyn[0]
                rlp1postemp = [
                    players.get(1).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(1).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlp2postemp = [
                    players.get(2).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(2).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlworldp1 = pixel_to_3d(rlp1postemp, homography, reference_points_3d)
                rlworldp2 = pixel_to_3d(rlp2postemp, homography, reference_points_3d)
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

            if len(ballxy) > 0:
                balltext = f"Ball position: {ballxy[-1][0]},{ballxy[-1][1]}"
                rlball = pixel_to_3d(
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
            print(f"finished writing frame {frame_count}")

            def csvwrite():
                with open("output/final.csv", "a") as f:
                    csvwriter = csv.writer(f)
                    # going to be putting the framecount, playerkeypoints, ball position, time, type of shot, and also match in play
                    running_frame / fps
                    shot = (
                        "None"
                        if type_of_shot is None
                        else type_of_shot[0] + " " + type_of_shot[1]
                    )
                    data = [
                        running_frame,
                        players.get(1).get_latest_pose().xyn[0],
                        players.get(2).get_latest_pose().xyn[0],
                        mainball.getloc(),
                        shot,
                    ]
                    csvwriter.writerow(data)

            # print(past_ball_pos)
            if past_ball_pos is not None:
                text = f"ball in rlworld: {pixel_to_3d([past_ball_pos[-1][0],past_ball_pos[-1][1]], homography, reference_points_3d)}"
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            try:
                csvwrite()
            except Exception:
                pass
            if running_frame > end:
                try:
                    with open("final.txt", "a") as f:
                        f.write(
                            csvanalyze.parse_through(csvstart, end, "output/final.csv")
                        )
                    csvstart = end
                    end += 100
                except Exception as e:
                    print(f"error: {e}")
                    pass
            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)

            print(f"finished frame {frame_count}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"error2: {e}")
        print(f"line was {e.__traceback__.tb_lineno}")
        print(f"other into about e: {e.__traceback__}")
        print(f"other info about e: {e.__traceback__.tb_frame}")
        print(f"other info about e: {e.__traceback__.tb_next}")
        print(f"other info about e: {e.__traceback__.tb_lasti}")


reference_points = []


def get_reference_points(path, frame_width, frame_height):
    # Mouse callback function to capture click events
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append((x, y))
            print(f"Point captured: ({x}, {y})")
            cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Court", frame1)

    # Function to save reference points to a file
    def save_reference_points(file_path):
        with open(file_path, "w") as f:
            json.dump(reference_points, f)
        print(f"reference points saved to {file_path}")

    # Function to load reference points from a file
    def load_reference_points(file_path):
        global reference_points
        with open(file_path, "r") as f:
            reference_points = json.load(f)
        print(f"reference points loaded from {file_path}")

    # Load the frame (replace 'path_to_frame' with the actual path)
    if os.path.isfile("reference_points.json"):
        load_reference_points("reference_points.json")
        print(f"Loaded reference points: {reference_points}")
        return reference_points
    else:
        print(
            "No reference points file found. Please click on the court to set reference points."
        )
        frame_count = 1
        cap2 = cv2.VideoCapture(path)
        frame1 = None
        while cap2.isOpened():
            success, frame = cap2.read()
            if not success:
                break
            frame_count += 1
            if frame_count == 100:
                frame1 = frame
                break
        frame1 = cv2.resize(frame1, (frame_width, frame_height))
        cv2.imshow("Court", frame1)
        cv2.setMouseCallback("Court", click_event)

        print(
            "Click on the key points of the court. Press 's' to save and 'q' to quit.\nMake sure to click in the following order shown by the example"
        )
        example_image = cv2.imread("output/annotated-squash-court.png")
        example_image_resized = cv2.resize(example_image, (frame_width, frame_height))
        cv2.imshow("Court Example", example_image_resized)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_reference_points("reference_points.json")
                cv2.destroyAllWindows()
                return reference_points
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return reference_points


def create_norfair_tracker():
    return Tracker(
        distance_function="euclidean",
        distance_threshold=50,
        hit_counter_max=10,
        filter_factory=OptimizedKalmanFilterFactory(),
        past_detections_length=5,
    )


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


def find_last_one(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 1:
            possibleis.append(i)
    # print(possibleis)
    if len(possibleis) > 1:
        return possibleis[-1]

    return -1


def cleanwrite():
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
    with open("output/final.json", "w") as f:
        f.write("[")
    with open("output/final.csv", "w") as f:
        f.write(
            "Frame count,Player 1 Keypoints,Player 2 Keypoints,Ball Position,Shot Type\n"
        )


def get_data(length):
    # go through the data in the file output/final.csv and then return all the data in a list
    for i in range(0, length):
        with open("output/final.csv", "r") as f:
            data = f.read()
    return data


def find_last_two(array):
    possibleis = []
    for i in range(len(array)):
        if array[i][1] == 2:
            possibleis.append(i)
    if len(possibleis) > 1:
        return possibleis[-1]
    return -1


def find_last(i, other_track_ids):
    possibleits = []
    for it in range(len(other_track_ids)):
        if other_track_ids[it][1] == i:
            possibleits.append(it)
    return possibleits[-1]


def pixel_to_3d_pixel_reference(pixel_point, pixel_reference, reference_points_3d):
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
def generate_homography(points_2d, points_3d):
    """
    Generate homography matrix from 2D pixel coordinates to 3D real world coordinates
    Args:
        points_2d: List of [x,y] pixel coordinates
        points_3d: List of [x,y,z] real world coordinates in meters
    Returns:
        3x3 homography matrix
    """
    # Convert to numpy arrays
    src_pts = np.array(points_2d, dtype=np.float32)
    dst_pts = np.array(points_3d, dtype=np.float32)

    # Remove y coordinate from 3D points since we're working on court plane
    dst_pts = np.delete(dst_pts, 1, axis=1)

    # Calculate homography matrix
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

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


# from squash.framepose import framepose

"""
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor
feature_extractor = FeatureExtractor(
    model_name='osnet_x1_0',  # You can choose other models as well
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
known_players_features = {}

def appearance_reid(player_crop, known_players_features, threshold=0.7):
    # Resize and preprocess the player crop
    player_crop_resized = cv2.resize(player_crop, (128, 256))
    # Convert BGR to RGB
    player_crop_rgb = cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = feature_extractor([player_crop_rgb])  # Returns a numpy array
    features = torch.tensor(features)
    features = F.normalize(features, p=2, dim=1)
    
    max_similarity = 0
    matched_player_id = None
    
    # Compare with known players
    for player_id, known_feature in known_players_features.items():
        similarity = F.cosine_similarity(features, known_feature).item()
        if similarity > max_similarity:
            max_similarity = similarity
            matched_player_id = player_id
    
    # Check if similarity exceeds the threshold
    if max_similarity > threshold:
        return matched_player_id, features
    else:
        return None, features
    
"""


"""
HISTOGRAM:::
import cv2
import numpy as np

# Initialize known player histograms with IDs 1 and 2
known_players_histograms = {}

def appearance_reid(player_crop, known_players_histograms, threshold=0.7):
    # Resize the player crop for consistency
    player_crop_resized = cv2.resize(player_crop, (64, 128))
    
    # Convert to HSV color space
    hsv_crop = cv2.cvtColor(player_crop_resized, cv2.COLOR_BGR2HSV)
    
    # Compute color histogram
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    
    max_similarity = 0
    matched_player_id = None
    
    # Compare with known players
    for player_id in [1, 2]:
        known_hist = known_players_histograms.get(player_id)
        if known_hist is not None:
            similarity = cv2.compareHist(hist, known_hist, cv2.HISTCMP_CORREL)
            if similarity > max_similarity:
                max_similarity = similarity
                matched_player_id = player_id
    
    # Check if similarity exceeds the threshold
    if max_similarity > threshold:
        return matched_player_id, hist
    else:
        # Assign the player to an available ID (1 or 2)
        available_ids = [1, 2]
        for pid in available_ids:
            if pid not in known_players_histograms:
                known_players_histograms[pid] = hist
                return pid, hist
        # If both IDs are taken but no match, default to the closest match
        return matched_player_id, hist
"""


def framepose(
    pose_model,
    frame,
    other_track_ids,
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
    occluded=False,
    importantdata=[],
):
    global known_players_features
    try:
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
            print(f"number of players found: {len(track_ids)}")
            # occluded as [[players_found, last_pos_p1, last_pos_p2, frame_number]...]
            if len(track_ids) < 2:
                print(player_last_positions)
                last_pos_p1 = player_last_positions.get(1, (None, None))
                last_pos_p2 = player_last_positions.get(2, (None, None))
                # print(f"last pos p1: {last_pos_p1}")
                occluded = []
                try:
                    occluded.append(
                        [
                            len(track_ids),
                            last_pos_p1,
                            last_pos_p2,
                            frame_count,
                        ]
                    )
                except Exception:
                    pass
            if len(track_ids) > 2:
                print(f"track ids were greater than 2: {track_ids}")
                return [
                    pose_model,
                    frame,
                    other_track_ids,
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
                    occluded,
                    importantdata,
                ]

            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x, y, w, h = box
                player_crop = frame[int(y) : int(y + h), int(x) : int(x + w)]

                if player_crop.size == 0:
                    continue
                if not find_match_2d_array(other_track_ids, track_id):
                    # player 1 has been updated last
                    if updated[0][1] > updated[1][1]:
                        if len(references2) > 1:
                            other_track_ids.append([track_id, 2])
                            print(f"added track id {track_id} to player 2")
                    else:
                        other_track_ids.append([track_id, 1])
                        print(f"added track id {track_id} to player 1")
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
                    playerid = 2
                    # player 1 was updated last
                elif updated[0][1] < updated[1][1]:
                    playerid = 1
                    # player 2 was updated last
                elif updated[0][1] == updated[1][1]:
                    playerid = 1
                    # both players were updated at the same time, so we are assuming that player 1 is the next player
                else:
                    print(f"could not find player id for track id {track_id}")
                    continue
                if track_id == 1:
                    references1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    references1[-1]
                    sum_pixels_in_bbox(frame, [x, y, w, h])
                    if len(references1) > 1 and len(references2) > 1:
                        if len(pixdiffs) < 5:
                            pixdiffs.append(abs(references1[-1] - references2[-1]))
                elif track_id == 2:
                    references2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    references2[-1]
                    sum_pixels_in_bbox(frame, [x, y, w, h])
                    if len(references1) > 1 and len(references2) > 1:
                        if len(pixdiffs) < 5:
                            pixdiffs.append(abs(references1[-1] - references2[-1]))
                # If player is already tracked, update their info
                if playerid in players:
                    players[playerid].add_pose(kp)
                    player_last_positions[playerid] = (x, y)  # Update position
                    players[playerid].add_pose(kp)
                    if playerid == 1:
                        updated[0][0] = True
                        updated[0][1] = frame_count
                    if playerid == 2:
                        updated[1][0] = True
                        updated[1][1] = frame_count
                if len(players) < max_players:
                    players[other_track_ids[track_id][0]] = Player(
                        player_id=other_track_ids[track_id][1]
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
                            cv2.circle(
                                annotated_frame, (int(x), int(y)), 3, (0, 0, 255), 5
                            )
                        else:
                            cv2.circle(
                                annotated_frame, (int(x), int(y)), 3, (255, 0, 0), 5
                            )
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
            # in the form of [ball shot type, player1 proximity to the ball, player2 proximity to the ball, ]
            importantdata = []
        return [
            pose_model,
            frame,
            other_track_ids,
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
            occluded,
            importantdata,
        ]
    except Exception as e:
        print(f"e: {e}")
        return [
            pose_model,
            frame,
            other_track_ids,
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
            occluded,
            importantdata,
        ]


# from squash import inferenceslicing
# from squash import deepsortframepose
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
    other_track_ids,
    updated,
    references1,
    references2,
    pixdiffs,
    players,
    player_last_positions,
    occluded,
    importantdata,
):
    try:
        highestconf = 0
        x1 = x2 = y1 = y2 = 0
        # Ball detection
        ball = ballmodel(frame)
        label = ""
        try:
            # Get the last 10 positions
            start_idx = max(0, len(past_ball_pos) - 10)
            positions = past_ball_pos[start_idx:]
            # Loop through positions
            for i in range(len(positions)):
                pos = positions[i]
                # Draw circle with constant radius
                radius = 5  # Adjust as needed
                cv2.circle(
                    annotated_frame, (pos[0], pos[1]), radius, (255, 255, 255), 2
                )
                # Draw line to next position
                if i < len(positions) - 1:
                    next_pos = positions[i + 1]
                    cv2.line(
                        annotated_frame,
                        (pos[0], pos[1]),
                        (next_pos[0], next_pos[1]),
                        (255, 255, 255),
                        2,
                    )
        except Exception:
            pass
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
                mainball.update(avg_x, avg_y, size)
                past_ball_pos.append([avg_x, avg_y, running_frame])
                math.hypot(
                    avg_x - mainball.getlastpos()[0], avg_y - mainball.getlastpos()[1]
                )
                drawmap(
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
            other_track_ids=other_track_ids,
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
            occluded=occluded,
            importantdata=importantdata,
        )
        other_track_ids = framepose_result[2]
        updated = framepose_result[3]
        references1 = framepose_result[4]
        references2 = framepose_result[5]
        pixdiffs = framepose_result[6]
        players = framepose_result[7]
        player_last_positions = framepose_result[9]
        annotated_frame = framepose_result[12]
        occluded = framepose_result[13]
        importantdata = framepose_result[14]
        # in the form of [ball shot type, player1 proximity to the ball, player2 proximity to the ball, ]
        importantdata.append(shot_type(past_ball_pos))
        importantdata.append(
            math.hypot(
                avg_x - player_last_positions.get(1)[0],
                avg_y - player_last_positions.get(1)[1],
            )
        )
        importantdata.append(
            math.hypot(
                avg_x - player_last_positions.get(2)[0],
                avg_y - player_last_positions.get(2)[1],
            )
        )
        # print(f'idata: {importantdata}')
        del framepose_result
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
            other_track_ids,  # 9
            updated,  # 10
            references1,  # 11
            references2,  # 12
            pixdiffs,  # 13
            players,  # 14
            player_last_positions,  # 15
            occluded,  # 16
            importantdata,  # 17
        ]
    except Exception:
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
            other_track_ids,  # 9
            updated,  # 10
            references1,  # 11
            references2,  # 12
            pixdiffs,  # 13
            players,  # 14
            player_last_positions,  # 15
            occluded,  # 16
            importantdata,  # 17
        ]


def reorganize_shots(alldata, min_sequence=5):
    if not alldata:
        return []

    # Filter out None types and replace with "unknown"
    cleaned_data = [
        [shot[0] if shot[0] is not None else "unknown"] + shot[1:] for shot in alldata
    ]

    # Step 1: Group consecutive shots
    sequences = []
    current_type = cleaned_data[0][0]
    current_sequence = []

    for shot in cleaned_data:
        if shot[0] == current_type:
            current_sequence.append(shot)
        else:
            if len(current_sequence) > 0:
                sequences.append((current_type, current_sequence))
            current_type = shot[0]
            current_sequence = [shot]
    if current_sequence:
        sequences.append((current_type, current_sequence))

    # Step 2: Fix short sequences, especially for crosscourts
    i = 0
    while i < len(sequences):
        shot_type = sequences[i][0]
        if len(sequences[i][1]) < min_sequence and shot_type.lower() == "crosscourt":
            short_sequence = sequences[i][1]

            # Look for same shot type in adjacent sequences
            borrowed_shots = []

            # Check forward and backward for shots to borrow
            for j in range(i + 1, len(sequences)):
                if sequences[j][0] == shot_type:
                    available = len(sequences[j][1])
                    needed = min_sequence - len(short_sequence) - len(borrowed_shots)
                    if available > min_sequence:
                        borrowed = sequences[j][1][:needed]
                        sequences[j] = (sequences[j][0], sequences[j][1][needed:])
                        borrowed_shots.extend(borrowed)
                        if len(borrowed_shots) + len(short_sequence) >= min_sequence:
                            break

            if len(borrowed_shots) + len(short_sequence) < min_sequence:
                for j in range(i - 1, -1, -1):
                    if sequences[j][0] == shot_type:
                        available = len(sequences[j][1])
                        needed = (
                            min_sequence - len(short_sequence) - len(borrowed_shots)
                        )
                        if available > min_sequence:
                            borrowed_prev = sequences[j][1][-needed:]
                            sequences[j] = (sequences[j][0], sequences[j][1][:-needed])
                            borrowed_shots = borrowed_prev + borrowed_shots
                            if (
                                len(borrowed_shots) + len(short_sequence)
                                >= min_sequence
                            ):
                                break

            # Update sequence with borrowed shots
            if len(borrowed_shots) + len(short_sequence) >= min_sequence:
                sequences[i] = (shot_type, borrowed_shots + short_sequence)
            else:
                # Merge with adjacent sequence if can't borrow enough
                if i > 0:
                    sequences[i - 1] = (
                        sequences[i - 1][0],
                        sequences[i - 1][1] + short_sequence,
                    )
                    sequences.pop(i)
                    i -= 1
                elif i < len(sequences) - 1:
                    sequences[i + 1] = (
                        sequences[i + 1][0],
                        short_sequence + sequences[i + 1][1],
                    )
                    sequences.pop(i)
                    i -= 1
        i += 1

    # Step 3: Format output
    result = []
    for shot_type, sequence in sequences:
        coords = []
        for shot in sequence:
            coords.extend([shot[1], shot[2]])
        result.append([shot_type, len(sequence), coords])

    return result


def visualize_shots(shot_data, width=640, height=360):
    # Create white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define color mapping (BGR format for OpenCV)
    colors = {
        "straight drive": (0, 0, 255),  # Red
        "crosscourt drive": (0, 255, 0),  # Green
        "crosscourt": (0, 255, 0),  # Green
        "unknown": (255, 0, 0),  # Blue
    }

    # Find coordinate ranges for normalization
    all_x = []
    all_y = []
    for sequence in shot_data:
        coords = sequence[2]
        for i in range(0, len(coords), 2):
            all_x.append(coords[i])
            all_y.append(coords[i + 1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding to prevent edge cases
    padding = 50

    # Draw circles and connecting lines for each shot sequence
    for sequence in shot_data:
        shot_type = sequence[0]
        coords = sequence[2]
        color = colors.get(shot_type.lower(), (255, 0, 0))

        prev_point = None
        # Process coordinates in pairs
        for i in range(0, len(coords), 2):
            # Normalize coordinates to fit window with padding
            x = int(
                ((coords[i] - x_min) / (x_max - x_min)) * (width - 2 * padding)
                + padding
            )
            y = int(
                ((coords[i + 1] - y_min) / (y_max - y_min)) * (height - 2 * padding)
                + padding
            )

            # Ensure coordinates are within bounds
            x = min(max(x, padding), width - padding)
            y = min(max(y, padding), height - padding)

            # Draw circle
            cv2.circle(image, (x, y), 3, color, -1)  # Increased radius to 3

            # Draw connecting line to previous point
            if prev_point is not None:
                cv2.line(image, prev_point, (x, y), color, 1)
            prev_point = (x, y)

            # Add shot type label at first point
            if i == 0:
                cv2.putText(
                    image,
                    shot_type,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

    return image


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


frame_height = 360
frame_width = 640


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
            # print(f"{threshballpos[i]}")
            # print(f'maxheight: {maxheight}')
            # print(f'threshballpos[i][1]: {threshballpos[i][1]}')
    if maxheight < (frame_height) / 1.35:
        height += "lob"
        # print(f'max height was {maxheight} and thresh was {(1.5*frame_height)/2}')
    else:
        height += "drive"
    return typeofshot + " " + height


def is_camera_angle_switched(frame, reference_image, threshold=0.5):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(reference_image_gray, frame_gray, full=True)
    return score < threshold


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
    except Exception:
        # print(
        #     f"got exception in is_match_in_play: {e}, line was {e.__traceback__.tb_lineno}"
        # )
        return False


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


def classify_shot(past_ball_pos, court_width=640, court_height=360):
    """
    Classifies shots based on trajectory angles and court position
    Returns [direction, shot_type, wall_hits]
    """
    try:
        if len(past_ball_pos) < 3:
            return ["straight", "drive", 0]
        # shorten the past ball positions to the last 5
        if len(past_ball_pos) > 5:
            past_ball_pos = past_ball_pos[-5:]
        # Calculate angles between consecutive points
        angles = []
        for i in range(len(past_ball_pos) - 2):
            x1, y1, _ = past_ball_pos[i]
            x2, y2, _ = past_ball_pos[i + 1]
            x3, y3, _ = past_ball_pos[i + 2]

            angle = calculate_angle(x1, y1, x2, y2, x3, y3)
            angles.append(angle)

        # Calculate horizontal displacement
        start_x = past_ball_pos[0][0]
        end_x = past_ball_pos[-1][0]
        displacement = abs(end_x - start_x) / court_width

        # Initialize variables
        direction = "straight"
        shot_type = "drive"
        wall_hits = count_wall_hits(past_ball_pos)

        # Classify direction based on displacement and angles
        if displacement > 0.4:
            direction = "wide_crosscourt"
        elif displacement > 0.25:
            direction = "crosscourt"
        elif displacement > 0.15:
            direction = "slight_crosscourt"
        elif displacement < 0.1:
            direction = "tight_straight"
        else:
            direction = "straight"

        # Classify shot type based on vertical movement
        avg_angle = sum(angles) / len(angles)
        if avg_angle > 60:
            shot_type = "lob"
        else:
            shot_type = "drive"

        return [direction, shot_type, wall_hits]

    except Exception:
        return ["straight", "drive", 0]


def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate angle between three points
    """

    # Calculate vectors
    vector1 = [x2 - x1, y2 - y1]
    vector2 = [x3 - x2, y3 - y2]

    # Calculate dot product
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitudes
    mag1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    mag2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate angle in degrees
    if mag1 * mag2 == 0:
        return 0

    angle = math.degrees(math.acos(dot_product / (mag1 * mag2)))
    return angle


def count_wall_hits(past_ball_pos, threshold=10):
    """
    Count number of wall hits based on direction changes
    """
    wall_hits = 0
    for i in range(1, len(past_ball_pos) - 1):
        x1, y1, _ = past_ball_pos[i - 1]
        x2, y2, _ = past_ball_pos[i]
        x3, y3, _ = past_ball_pos[i + 1]

        if abs(x2 - x1) < threshold and abs(x3 - x2) < threshold:
            wall_hits += 1
    return wall_hits


def is_ball_false_pos(past_ball_pos, speed_threshold=50, angle_threshold=45):
    if len(past_ball_pos) < 3:
        return False  # Not enough data to determine

    x0, y0, frame0 = past_ball_pos[-3]
    x1, y1, frame1 = past_ball_pos[-2]
    x2, y2, frame2 = past_ball_pos[-1]

    # Speed check
    time_diff = frame2 - frame1
    if time_diff == 0:
        return True  # Same frame, likely false positive

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    speed = distance / time_diff

    if speed > speed_threshold:
        return True  # Speed exceeds threshold, likely false positive

    # Angle check
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)

    def compute_angle(v1, v2):
        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0  # Cannot compute angle with zero-length vector
        cos_theta = dot_prod / (mag1 * mag2)
        cos_theta = max(-1, min(1, cos_theta))  # Clamp to [-1,1]
        angle = math.degrees(math.acos(cos_theta))
        return angle

    angle_change = compute_angle(v1, v2)

    if angle_change > angle_threshold:
        return True  # Sudden angle change, possible false positive

    return False


def predict_next_pos(past_ball_pos, num_predictions=2):
    # Define a fixed sequence length
    max_sequence_length = 10

    # Prepare the positions array
    data = np.array(past_ball_pos)
    positions = data[:, :2]  # Extract x and y coordinates

    # Ensure positions array has the fixed sequence length
    if positions.shape[0] < max_sequence_length:
        padding = np.zeros((max_sequence_length - positions.shape[0], 2))
        positions = np.vstack((padding, positions))
    else:
        positions = positions[-max_sequence_length:]

    # Prepare input for LSTM
    X_input = positions.reshape((1, max_sequence_length, 2))

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(max_sequence_length, 2)))
    model.add(Dense(2))

    # Load pre-trained model weights if available
    # model.load_weights('path_to_weights.h5')

    predictions = []
    input_seq = X_input.copy()
    for _ in range(num_predictions):
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0])

        # Update input sequence
        input_seq = np.concatenate((input_seq[:, 1:, :], pred.reshape(1, 1, 2)), axis=1)
    return predictions


def input_model(csvdata):
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {
            "role": "system",
            "content": 'You are a squash coach. You are to read through this csv data structured in the format: "Frame count,Player 1 Keypoints,Player 2 Keypoints,Ball Position,Shot Type" and provide a response that summarizes what happened',
        },
        {
            "role": "user",
            "content": f"Here is an example response: In frame 66, Player 1 is positioned in the back right quarter of the court, while Player 2 is in the front left quarter. Player 1 hits a crosscourt drive, and the ball was successfully hit, bouncing off 1 wall. Here is the data, reply back in the same way as the example: {csvdata}",
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    try:
        main()
    # get keyboarinterrupt error
    except KeyboardInterrupt:
        print("keyboard interrupt")

        exit()
    except Exception:
        # print(f"error: {e}")
        pass
