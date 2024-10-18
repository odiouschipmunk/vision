import cv2
from ultralytics import YOLO
import numpy as np
import math
from squash import Refrencepoints, Predict, Functions
pose_model = YOLO("models/yolo11m-pose.pt")
ballmodel = YOLO("trained-models/g-ball2.pt")
video_file = "Squash Farag v Hesham - Houston Open 2022 - Final Highlights.mp4"
video_folder = "full-games"
path = "main.mp4"
import matplotlib.pyplot as plt
ballvideopath='output/balltracking.mp4'
cap = cv2.VideoCapture(path)
frame_width = 640
frame_height = 360
players = {}
courtref = 0
occlusion_times = {}
last_frame = []
for i in range(1, 3):
    occlusion_times[i] = 0
from squash.Ball import Ball
import logging
from squash.Player import Player
max_players = 2
player_last_positions = {}
frame_count = 0
trackid1 = True
trackid2 = True
logging.getLogger("ultralytics").setLevel(logging.ERROR)
output_path = "annotated.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file
fps = 25  # Frames per second
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
ball_out = cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))
avgp1ref=avgp2ref=0
allp1refs=allp2refs=[]
mainball = Ball(0, 0, 0, 0)
ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
playerRefrence1 = 0
playerRefrence2 = 0
otherTrackIds = [[0, 0], [1, 1], [2, 2]]
updated = [[False, 0], [False, 0]]
refrence_points=Refrencepoints.get_refrence_points(path=path, frame_width=frame_width, frame_height=frame_height)
refrences1 = []
refrences2 = []
diff = []
diffTrack = False
updateref=True
p1ref=p2ref=0
p1embeddings = [[]]
p2embeddings = [[]]
pixdiffs=[]
pixdiff1percentage=pixdiff2percentage=[]
avgcosinediff=0
cosinediffs=[]
from PIL import Image
player1imagerefrence=player2imagerefrence=None
player1refrenceembeddings=player2refrenceembeddings=None
p1distancesfromT = []
p2distancesfromT = []
player_move = [[]]
courtref = np.int64(courtref)
refrenceimage = None
theatmap1 = np.zeros((frame_height, frame_width), dtype=np.float32)
theatmap2 = np.zeros((frame_height, frame_width), dtype=np.float32)
outlierdiffs=[]
heatmap_overlay_path='output/white.png'
heatmap_image=cv2.imread(heatmap_overlay_path)
if heatmap_image is None:
    raise FileNotFoundError(f'Could not find heatmap overlay image at {heatmap_overlay_path}')
heatmap_ankle=np.zeros_like(heatmap_image, dtype=np.float32)
ballxy=[]
running_frame=0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    frame_count += 1
    if frame_count == 1:
        print("frame 1")
        courtref = np.int64(
            Functions.sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
        )
        print(courtref)
        refrenceimage = frame
    if frame_count>=250:
        break
    running_frame+=1
    if running_frame>=500:
        updatedref=False
    if len(refrences1) !=0 and len(refrences2)!=0:
        avgp1ref=sum(refrences1)/len(refrences1)
        avgp2ref=sum(refrences2)/len(refrences2)
    if Functions.is_camera_angle_switched(frame, refrenceimage, threshold=0.6):
        print("camera angle switched")
        continue
    currentref = int(Functions.sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height]))
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
    pose_results = pose_model(frame)
    annotated_frame = pose_results[0].plot()
    ballframe = frame.copy()
    for refrence in refrence_points:
        cv2.circle(frame, refrence, 5, (0, 255, 0), -1)
    if (
        pose_results[0].keypoints.xyn is not None
        and len(pose_results[0].keypoints.xyn[0]) > 0
    ):
        for person in pose_results[0].keypoints.xyn:

            if len(person) >= 17: 

                left_ankle_x = int(
                    person[16][0] * frame_width
                )  # Scale the X coordinate
                left_ankle_y = int(
                    person[16][1] * frame_height
                )  # Scale the Y coordinate
                right_ankle_x = int(
                    person[15][0] * frame_width
                )  # Scale the X coordinate
                right_ankle_y = int(
                    person[15][1] * frame_height
                )  # Scale the Y coordinate
                
    else:
        # print("No keypoints detected in this frame.")
        continue
    highestconf = 0
    x1 = x2 = y1 = y2 = 0
    # Ball detection
    # make it so that if it detects the ball in the same place multiple times it takes that out
    label = ""
    for box in ball[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = ballmodel.names[int(box.cls)]
        confidence = float(box.conf)  # Convert tensor to float
        avgxtemp = int((x1temp + x2temp) / 2)
        avgytemp = int((y1temp + y2temp) / 2)
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
    cv2.circle(ballframe, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 255, 0), -1)
    avg_x = int((x1 + x2) / 2)
    avg_y = int((y1 + y2) / 2)
    distance = 0
    size = avg_x * avg_y
    if avg_x > 0 or avg_y > 0:
        if mainball.getlastpos()[0] != avg_x or mainball.getlastpos()[1] != avg_y:
            # print(mainball.getlastpos())
            # print(mainball.getloc())
            mainball.update(avg_x, avg_y, size)
            # print(mainball.getlastpos())
            # print(mainball.getloc())
            distance = math.hypot(
                avg_x - mainball.getlastpos()[0], avg_y - mainball.getlastpos()[1]
            )

            with open("output/ball.txt", "a") as f:
                f.write(
                    f"Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n"
                )
                # print(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
                Functions.drawmap(
                    mainball.getloc()[0],
                    mainball.getloc()[1],
                    mainball.getlastpos()[0],
                    mainball.getlastpos()[1],
                    ballmap,
                )

    '''
    FRAMEPOSE
    '''
    
    track_results = pose_model.track(frame, persist=True)
    try:
        if (
            track_results
            and hasattr(track_results[0], "keypoints")
            and track_results[0].keypoints is not None
        ):
            # Extract boxes, track IDs, and keypoints from pose results
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()

            current_ids = set(track_ids)

            # Update or add players for currently visible track IDs
            # note that this only works with occluded players < 2, still working on it :(

            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x, y, w, h = box
                player_crop = frame[int(y):int(y+h), int(x):int(x+w)]
                player_image = Image.fromarray(player_crop)
                #embeddings=get_image_embeddings(player_image)
                psum=Functions.sum_pixels_in_bbox(frame, [x, y, w, h])
                if not Functions.find_match_2d_array(otherTrackIds, track_id):
                    # player 1 has been updated last
                    if updated[0][1] > updated[1][1]:
                        if len(refrences2)>1:
                            #comparing it to itself, if percentage is greater than 75, then its probably a different player
                            if (100*abs(psum-refrences2[-1])/psum)>75:
                                otherTrackIds.append([track_id, 1])
                                print(f"added track id {track_id} to player 1 using image refrences, as image similarity was {100*abs(psum-refrences2[-1])/psum}")
                            else: 
                                otherTrackIds.append([track_id, 2])
                                print(f"added track id {track_id} to player 2")
                    else:
                            if (100*abs(psum-refrences1[-1])/psum)>75:
                                otherTrackIds.append([track_id, 2])
                                print(f"added track id {track_id} to player 2 using image refrences, as image similarity was {100*abs(psum-refrences1[-1])/psum}")
                            else:
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
                    if len(refrences2)>1:
                            #comparing it to itself, if percentage is greater than 75, then its probably a different player
                            if (100*abs(psum-refrences2[-1])/psum)>75:
                                playerid = 1
                            else:
                                playerid = 2
                    else:
                        playerid=2
                    # player 1 was updated last
                elif updated[0][1] < updated[1][1]:
                    if len(refrences1)>1:
                            #comparing it to itself, if percentage is greater than 75, then its probably a different player
                            if (100*abs(psum-refrences1[-1])/psum)>75:
                                playerid = 2
                            else:
                                playerid = 1
                    else:
                        playerid=1
                    # player 2 was updated last
                elif updated[0][1] == updated[1][1]:
                    if len(refrences1)>1 and len(refrences2)>1:
                        if (100*abs(psum-refrences1[-1])/psum) > (100*abs(psum-refrences2[-1])/psum):
                            playerid = 2
                        else:
                            playerid=1
                    else:
                        playerid = 1
                    # both players were updated at the same time, so we are assuming that player 1 is the next player
                else:
                    print(f'could not find player id for track id {track_id}')
                    continue


                # player refrence appending for maybe other stuff
                #using track_id and not playerid so that it is definitely the correct player
                #maybe use playerid instead of track_id later on, but for right now its fine tbh
                if track_id == 1:
                    refrences1.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
                    temp1 = refrences1[-1]
                    p1ref=Functions.sum_pixels_in_bbox(frame, [x, y, w, h])
                    
                    #p1embeddings.append(embeddings)



                    if (len(refrences1) >1 and len(refrences2)>1):
                        if len(pixdiffs)<5:
                            pixdiffs.append(abs(refrences1[-1]-refrences2[-1]))
                       

                                
                    if player1imagerefrence is None:
                        player1imagerefrence=player_image
                        #player1refrenceembeddings=embeddings
                    #bookmark for pixel differences and cosine similarity
                    '''
                    
                        '''


                elif track_id == 2:
                    refrences2.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
                    temp2 = refrences2[-1]
                    p2ref=Functions.sum_pixels_in_bbox(frame, [x, y, w, h])
                    #print(f'p2ref: {p2ref}')
                    #print(embeddings.shape)
                    #p2embeddings.append(embeddings)

                    if (len(refrences1) >1 and len(refrences2)>1):
                        if len(pixdiffs)<5:
                            pixdiffs.append(abs(refrences1[-1]-refrences2[-1]))
                       

                # print(f'even though we are working with {otherTrackIds[track_id][0]}, the player id is {playerid}')
                #print(otherTrackIds)
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
    except Exception as e:
        print("GOT ERROR: ", e)
        pass




    # Save the heatmap
    # print(players)
    # print(players.get(1).get_latest_pose())
    # print(players.get(2).get_latest_pose())

    # print(len(players))
    if players.get(1) and players.get(2) is not None:
        if (
            players.get(1).get_latest_pose()
            and players.get(2).get_latest_pose() is not None
        ):
            p1x = (
                (
                    players.get(1).get_latest_pose().xyn[0][16][0]
                    + players.get(1).get_latest_pose().xyn[0][15][0]
                )
                / 2
            ) * frame_width
            p1y = (
                (
                    players.get(1).get_latest_pose().xyn[0][16][1]
                    + players.get(1).get_latest_pose().xyn[0][15][1]
                )
                / 2
            ) * frame_height
            p2x = (
                (
                    players.get(2).get_latest_pose().xyn[0][16][0]
                    + players.get(2).get_latest_pose().xyn[0][15][0]
                )
                / 2
            ) * frame_width
            p2y = (
                (
                    players.get(2).get_latest_pose().xyn[0][16][1]
                    + players.get(2).get_latest_pose().xyn[0][15][1]
                )
                / 2
            ) * frame_height

    
    # Display ankle positions of both players
    if players.get(1) and players.get(2) is not None:
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
            except Exception as e:
                p1_left_ankle_x = (
                    p1_left_ankle_y
                ) = p1_right_ankle_x = p1_right_ankle_y = 0
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
            except Exception as e:
                p2_left_ankle_x = (
                    p2_left_ankle_y
                ) = p2_right_ankle_x = p2_right_ankle_y = 0
            # Display the ankle positions on the bottom left of the frame
            avgxank1=int((p1_left_ankle_x+p1_right_ankle_x)/2)
            avgyank1=int((p1_left_ankle_y+p1_right_ankle_y)/2)
            avgxank2=int((p2_left_ankle_x+p2_right_ankle_x)/2)
            avgyank2=int((p2_left_ankle_y+p2_right_ankle_y)/2)
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
            '''
            # print(refrence_points)
            p1distancefromT = math.hypot(
                refrence_points[6][0] - avgpx1, refrence_points[6][1] - avgpy1
            )
            p2distancefromT = math.hypot(
                refrence_points[6][0] - avgpx2, refrence_points[6][1] - avgpy2
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
            plt2=plt
            plt.plot(p1distancesfromT, color='blue', label='P1 Distance from T')
            plt2.plot(p2distancesfromT, color='red', label='P2 Distance from T')

            # Add labels and title
            plt.xlabel('Time (frames)')
            plt2.xlabel('Time (frames)')
            plt.ylabel('Distance from T')
            plt2.ylabel('Distance from T')
            plt.title('Distance from T over Time')
            plt2.title('Distance from T over Time')
            plt.legend()
            plt2.legend()

            # Save the plot to a file
            plt.savefig('output/distance_from_t_over_time1.png')
            plt2.savefig('output/distance_from_t_over_time2.png')

            # Close the plot to free up memory
            plt.close()
            plt2.close()
            '''
    for ref in refrence_points:
        # cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(annotated_frame, (ref[0], ref[1]), 5, (0, 255, 0), 2)




    if players.get(1).get_latest_pose() is not None and players.get(2).get_latest_pose() is not None:
        player1anklex = players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
        player1ankley = players.get(1).get_latest_pose().xyn[0][16][1] * frame_height
        player2anklex = players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
        player2ankley = players.get(2).get_latest_pose().xyn[0][16][1] * frame_height

        # Draw points for player 1 (blue) and player 2 (red)
        cv2.circle(heatmap_image, (int(player1anklex), int(player1ankley)), 5, (255, 0, 0), -1)  # Blue points for player 1
        cv2.circle(heatmap_image, (int(player2anklex), int(player2ankley)), 5, (0, 0, 255), -1)  # Red points for player 2

        # Apply Gaussian blur to the heatmap to make it blurrier
        blurred_heatmap_ankle = cv2.GaussianBlur(heatmap_image, (51, 51), 0)

        # Normalize the heatmap to the range [0, 255]
        blurred_heatmap = cv2.normalize(blurred_heatmap_ankle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert the heatmap to uint8
        blurred_heatmap = blurred_heatmap.astype(np.uint8)

        # Apply color map to the heatmap
        heatmap_overlay = cv2.applyColorMap(blurred_heatmap, cv2.COLORMAP_JET)

        # Create a white image with the same dimensions as the heatmap
        white_image = np.ones_like(heatmap_overlay) * 255

        # Combine the images
        combined_image = cv2.addWeighted(white_image, 0.5, heatmap_overlay, 0.5, 0)

        # Save the combined image
        cv2.imwrite('output/heatmap_ankle.png', combined_image)


    
    ballx=bally=0
    #ball stuff
    if mainball is not None and mainball.getlastpos() is not None and mainball.getlastpos() != (0, 0):
        ballx = mainball.getlastpos()[0]
        bally = mainball.getlastpos()[1]
        if ballx != 0 and bally != 0:
            if [ballx, bally] not in ballxy:
                ballxy.append([ballx, bally, frame_count])
                print(f'ballx: {ballx}, bally: {bally}, appended to ballxy with length {len(ballxy)} and frame count as : {frame_count}')

    # Draw the ball trajectory
    if len(ballxy) > 2:
        for i in range(1, len(ballxy)):
            if ballxy[i - 1] is None or ballxy[i] is None:
                continue
            if ballxy[i][2] - ballxy[i - 1][2] < 7:
                if frame_count - ballxy[i][2] < 7:
                    cv2.line(annotated_frame, (ballxy[i - 1][0], ballxy[i - 1][1]), (ballxy[i][0], ballxy[i][1]), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (ballxy[i - 1][0], ballxy[i - 1][1]), 5, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, (ballxy[i][0], ballxy[i][1]), 5, (0, 255, 0), -1)
                    next_pos = Predict.predict_next_position((ballxy[i - 1][0], ballxy[i - 1][1]), (ballxy[i][0], ballxy[i][1]), frame_width, frame_height)
                    cv2.circle(annotated_frame, (next_pos[0], next_pos[1]), 5, (0, 255, 0), -1)

    for ball_pos in ballxy:
        if frame_count - ball_pos[2] < 7:
            #print(f'wrote to frame on line 1028 with coords: {ball_pos}')
            cv2.circle(annotated_frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 0), -1)

    ball_out.write(annotated_frame)
    out.write(annotated_frame)
    cv2.imshow("Annotated Frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()