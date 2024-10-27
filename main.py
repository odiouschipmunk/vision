

def main():
    """
    The `main` function processes video frames to detect players, their poses, and the ball trajectory
    in a squash game.
    """
    import cv2
    from ultralytics import YOLO
    import numpy as np
    import math
    from squash import Refrencepoints, Functions
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from squash.Ball import Ball
    import logging
    from squash.Player import Player
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_metric

    print("imported all")
    # Define the reference points in pixel coordinates (image)
    # These should be the coordinates of the reference points in the image
    # TODO: use embeddings to correctly find the different players
    ball_predict = tf.keras.models.load_model("ball_position_model(25k).keras")

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
    ballmodel = YOLO("trained-models/g-ball2.pt")
    # racketmodel=YOLO('trained-models/squash-racket.pt')
    # courtmodel=YOLO('trained-models/court-key!.pt')
    # Video file path
    video_file = "Squash Farag v Hesham - Houston Open 2022 - Final Highlights.mp4"
    video_folder = "full-games"
    path = "main.mp4"
    print("loaded models")
    ballvideopath = "output/balltracking.mp4"
    cap = cv2.VideoCapture(path)
    with open("output/final.txt", "a") as f:
        f.write(f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n")
    frame_width = 1920
    frame_height = 1080
    players = {}
    courtref = 0
    occlusion_times = {}
    last_frame = []
    for i in range(1, 3):
        occlusion_times[i] = 0
    future_predict = None
    # Get video dimensions
    biggestx=biggesty=smallestx=smallesty=0
    max_players = 2
    player_last_positions = {}
    frame_count = 0
    trackid1 = True
    trackid2 = True
    ball_false_pos=[]
    past_ball_pos=[]
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    output_path = "output/annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file
    fps = 25  # Frames per second
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    ball_out = cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))

    avgp1ref = avgp2ref = 0
    allp1refs = allp2refs = []

    def sum_pixels_in_bbox(frame, bbox):
        x, y, w, h = bbox
        roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
        return np.sum(roi, dtype=np.int64)

    # Create a blank canvas for heatmap based on video resolution

    mainball = Ball(0, 0, 0, 0)
    ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    playerRefrence1 = 0
    playerRefrence2 = 0
    # other track ids necessary as since players get occluded, im just going to assign that track id to the previous id(1 or 2) to the last occluded player
    # really need to fix this as if there are 2 occluded players, it will not work
    otherTrackIds = [[0, 0], [1, 1], [2, 2]]
    updated = [[False, 0], [False, 0]]
    refrence_points = []
    refrence_points = Refrencepoints.get_refrence_points(
        path=path, frame_width=frame_width, frame_height=frame_height
    )

    refrences1 = []
    refrences2 = []
    diff = []
    diffTrack = False
    updateref = True
    """
    def findRef(img):
        return cv2.
    """
    p1ref = p2ref = 0

    p1embeddings = [[]]
    p2embeddings = [[]]
    pixdiffs = []
    pixdiff1percentage = pixdiff2percentage = []
    avgcosinediff = 0
    cosinediffs = []

    player1imagerefrence = player2imagerefrence = None
    player1refrenceembeddings = player2refrenceembeddings = None

    p1distancesfromT = []
    p2distancesfromT = []

    player_move = [[]]
    courtref = np.int64(courtref)
    refrenceimage = None

    def is_camera_angle_switched(frame, refrence_image, threshold=0.5):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        refrence_image_gray = cv2.cvtColor(refrence_image, cv2.COLOR_BGR2GRAY)
        score, _ = ssim_metric(refrence_image_gray, frame_gray, full=True)
        return score < threshold

    # note for anyone else seeing this:
    # refrence[0] is x val and [1] is y val
    # refrence[0] is top left, 
    # refrence[1] is top right
    # refrence[2] is bottom right
    # refrence[3] is bottom left
    # refrence[4] is T
    # refrence[5] is left bottom of service box
    # refrence[6] is right bottom of service box
    # refrence[7] is left of service line
    # refrence[8] is right of service line
    # refrence[9] is left of the top line of the front court
    # refrence[10] is right of the top line of the front court
    reference_points_3d = [
        [0, 0, 5.64],       # Top-left corner
        [6.4, 0, 5.64],     # Top-right corner
        [6.4, 9.75, 0],     # Bottom-right corner
        [0, 9.75, 0],       # Bottom-left corner
        [3.2, 5.44, 0],     # "T" point
        [1.6, 5.44, 0],     # Left bottom of the service box
        [4.8, 5.44, 0],     # Right bottom of the service box
        [1.6, 0, 1.78],     # Left of the service line
        [4.8, 0, 1.78],     # Right of the service line
        [0, 0, 4.57],       # Left of the top line of the front court
        [6.4, 0, 4.57]      # Right of the top line of the front court
    ]    
    
    theatmap1 = np.zeros((frame_height, frame_width), dtype=np.float32)
    theatmap2 = np.zeros((frame_height, frame_width), dtype=np.float32)
    outlierdiffs = []
    heatmap_overlay_path = "output/white.png"
    heatmap_image = cv2.imread(heatmap_overlay_path)
    if heatmap_image is None:
        raise FileNotFoundError(
            f"Could not find heatmap overlay image at {heatmap_overlay_path}"
        )
    heatmap_ankle = np.zeros_like(heatmap_image, dtype=np.float32)

    ballxy = []

    homography=Functions.calculate_homography(refrence_points, reference_points_3d)
    
    running_frame = 0
    print("started video input")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        frame_count += 1

        if running_frame >= 500:
            updatedref = False
        if frame_count >= 250:
            cap.release()
            cv2.destroyAllWindows()
        if len(refrences1) != 0 and len(refrences2) != 0:
            avgp1ref = sum(refrences1) / len(refrences1)
            avgp2ref = sum(refrences2) / len(refrences2)
        running_frame += 1

        """
        if len(p1embeddings) != 0 and len(p2embeddings) != 0 and len(p1embeddings) > 1 and len(p2embeddings) > 1:
            #print(len(p1embeddings[-1]))
            #print(p1embeddings)
            similarity_p1 = cosine_similarity(p1embeddings[-1], p2embeddings[-1])
            similarity_p2 = cosine_similarity(p2embeddings[-1], p1embeddings[-1])
            print(f"Cosine Similarity p1: {similarity_p1}")
            print(f"Cosine Similarity p2: {similarity_p2}")
        """
        # if (biggestx == 0 or biggesty == 0 or smallestx == 0 or smallesty == 0) and len(refrence_points) == 11:
        #     biggestx=refrence_points[2][0]
        #     biggesty=refrence_points[9][1]
        #     smallestx=refrence_points[0][0]
        #     smallesty=refrence_points[3][1]
        #     frame=frame[smallesty:biggesty, smallestx:biggestx]
        
        if running_frame == 1:
            print("frame 1")
            courtref = np.int64(
                sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height])
            )
            print(courtref)
            refrenceimage = frame

        if is_camera_angle_switched(frame, refrenceimage, threshold=0.5):
            print("camera angle switched")
            continue

        # print(len(players))

        currentref = int(sum_pixels_in_bbox(frame, [0, 0, frame_width, frame_height]))

        # general court refrence to only get the first camera angle throughout the video
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
        #     biggestx = max(refrence_points[2][0], refrence_points[1][0])
        #     biggesty = max(refrence_points[3][1], refrence_points[2][1])
        #     smallestx = min(refrence_points[0][0], refrence_points[3][0])
        #     smallesty = min(refrence_points[9][1], refrence_points[10][1])
            
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
        #pose_results = pose_model(frame)
        # racket_results=racketmodel(frame)
        # only plot the top 2 confs
        annotated_frame = frame.copy()#pose_results[0].plot()
        
        # court_results=courtmodel(frame)
        # Check if keypoints exist and are not empty
        # print(pose_results)
        
        #false_pos=Functions.ball_is_false_positive(past_ball_pos)
        #if false_pos is not None:
            #ball_false_pos.append(false_pos)
            #print(f'ball false pos: {ball_false_pos}')
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

        avg_x = int((x1 + x2) / 2)
        avg_y = int((y1 + y2) / 2)
        distance = 0
        size = avg_x * avg_y
        if avg_x > 0 or avg_y > 0:
            if mainball.getlastpos()[0] != avg_x or mainball.getlastpos()[1] != avg_y:
                # print(mainball.getlastpos())
                # print(mainball.getloc())
                mainball.update(avg_x, avg_y, size)
                past_ball_pos.append([avg_x, avg_y, running_frame])
                # print(mainball.getlastpos())
                # print(mainball.getloc())
                distance = math.hypot(
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

        track_results = pose_model.track(frame, persist=True, show=False)
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
                    player_crop = frame[int(y) : int(y + h), int(x) : int(x + w)]
                    player_image = Image.fromarray(player_crop)
                    # embeddings=get_image_embeddings(player_image)
                    psum = sum_pixels_in_bbox(frame, [x, y, w, h])
                    if not Functions.find_match_2d_array(otherTrackIds, track_id):
                        # player 1 has been updated last
                        if updated[0][1] > updated[1][1]:
                            if len(refrences2) > 1:
                                # comparing it to itself, if percentage is greater than 75, then its probably a different player
                                # if (100 * abs(psum - refrences2[-1]) / psum) > 75:
                                #     otherTrackIds.append([track_id, 1])
                                #     print(
                                #         f"added track id {track_id} to player 1 using image refrences, as image similarity was {100*abs(psum-refrences2[-1])/psum}"
                                #     )
                                # else:
                                otherTrackIds.append([track_id, 2])
                                print(f"added track id {track_id} to player 2")
                        else:
                            # if (100 * abs(psum - refrences1[-1]) / psum) > 75:
                            #     otherTrackIds.append([track_id, 2])
                            #     print(
                            #         f"added track id {track_id} to player 2 using image refrences, as image similarity was {100*abs(psum-refrences1[-1])/psum}"
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
                        # if len(refrences2) > 1:
                        #     # comparing it to itself, if percentage is greater than 75, then its probably a different player
                        #     if (100 * abs(psum - refrences2[-1]) / psum) > 75:
                        #         playerid = 1
                        #     else:
                        #         playerid = 2
                        # else:
                        playerid = 2
                        # player 1 was updated last
                    elif updated[0][1] < updated[1][1]:
                        # if len(refrences1) > 1:
                        #     # comparing it to itself, if percentage is greater than 75, then its probably a different player
                        #     if (100 * abs(psum - refrences1[-1]) / psum) > 75:
                        #         playerid = 2
                        #     else:
                        #         playerid = 1
                        # else:
                        playerid = 1
                        # player 2 was updated last
                    elif updated[0][1] == updated[1][1]:
                        # if len(refrences1) > 1 and len(refrences2) > 1:
                        #     if (100 * abs(psum - refrences1[-1]) / psum) > (
                        #         100 * abs(psum - refrences2[-1]) / psum
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

                    # player refrence appending for maybe other stuff
                    # using track_id and not playerid so that it is definitely the correct player
                    # maybe use playerid instead of track_id later on, but for right now its fine tbh
                    if track_id == 1:
                        refrences1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                        temp1 = refrences1[-1]
                        p1ref = sum_pixels_in_bbox(frame, [x, y, w, h])

                        # p1embeddings.append(embeddings)

                        if len(refrences1) > 1 and len(refrences2) > 1:
                            if len(pixdiffs) < 5:
                                pixdiffs.append(abs(refrences1[-1] - refrences2[-1]))
                            """
                            USE FOR COSINE SIMILARITY BOOKMARK
                            else:
                                if abs(refrences1[-1]-refrences2[-1])>2*sum(pixdiffs)/len(pixdiffs):
                                    print(f'probably too big of a difference between the two players, pix diff: {abs(refrences1[-1]-refrences2[-1])} with percentage as {100*abs(refrences1[-1]-refrences2[-1])/refrences1[-1]}')
                                else:
                                    print(f'pix diff: {abs(refrences1[-1]-refrences2[-1])}')
                                    print(f'average pixel diff: {sum(pixdiffs)/(len(pixdiffs))}')
                                    pixdiff1percentage.append(100*abs(refrences1[-1]-refrences2[-1])/refrences1[-1])
                                    print(f'pixel diff in percentage for p1: {pixdiff1percentage[-1]}')
                                    print(f'largest percentage pixel diff: {max(pixdiff1percentage)}')
                                    print(f'smallest percentage pixel diff: {min(pixdiff1percentage)}')
                            """

                        if player1imagerefrence is None:
                            player1imagerefrence = player_image
                            # player1refrenceembeddings=embeddings
                        # bookmark for pixel differences and cosine similarity
                        """
                        if len(p2embeddings) > 1:
                            p1refrence=cosine_similarity(p1embeddings[-1], player1refrenceembeddings)
                            p1top2=cosine_similarity(p1embeddings[-1], p2embeddings[-1])
                            cosinediffs.append(abs(p1refrence-p1top2))
                            avgcosinediff=sum(cosinediffs)/len(cosinediffs)
                            
                            print(f'average cosine difference: {avgcosinediff}')       
                            print(f'highest cosine diff: {max(cosinediffs)}') 
                            print(f'lowest cosine diff: {min(cosinediffs)}')
                            print(f'this is player 1, with a cosine similarity of {p1refrence} to its refrence image')
                            print(f'this is player 1, with a cosine similarity of {p1top2} to player 2 right now')
                            print(f'difference between p1 refrence and p2 right now: {abs(p1refrence-p1top2)}')
                            """

                    elif track_id == 2:
                        refrences2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                        temp2 = refrences2[-1]
                        p2ref = sum_pixels_in_bbox(frame, [x, y, w, h])
                        # print(f'p2ref: {p2ref}')
                        # print(embeddings.shape)
                        # p2embeddings.append(embeddings)

                        if len(refrences1) > 1 and len(refrences2) > 1:
                            if len(pixdiffs) < 5:
                                pixdiffs.append(abs(refrences1[-1] - refrences2[-1]))
                            """
                            USE FOR COSINE SIMILARITY BOOKMARK
                            else:
                                if abs(refrences1[-1]-refrences2[-1])>2*sum(pixdiffs)/len(pixdiffs):
                                    print(f'probably too big of a difference between the two players with pix diff: {abs(refrences1[-1]-refrences2[-1])} with percentage as {100*abs(refrences2[-1]-refrences1[-1])/refrences2[-1]}')
                                else:
                                    print(f'pix diff: {abs(refrences1[-1]-refrences2[-1])}')
                                    print(f'average pixel diff: {sum(pixdiffs)/(len(pixdiffs))}')
                                    pixdiff1percentage.append(100*abs(refrences2[-1]-refrences1[-1])/refrences2[-1])
                                    print(f'pixel diff in percentage for p2: {pixdiff2percentage[-1]}')
                                    print(f'largest percentage pixel diff: {max(pixdiff2percentage)}')
                                    print(f'smallest percentage pixel diff: {min(pixdiff2percentage)}')
                            """

                        # print(p2embeddings)
                        """
                        if player2imagerefrence is None:
                            player2imagerefrence=player_image
                            player2refrenceembeddings=embeddings
                        


                        if len(p1embeddings) > 1:
                            p2refrence=cosine_similarity(p2embeddings[-1], player2refrenceembeddings)
                            p2top1=cosine_similarity(p2embeddings[-1], p1embeddings[-1])
                            
                            cosinediffs.append(abs(p2refrence-p2top1))
                            avgcosinediff=sum(cosinediffs)/len(cosinediffs)
                            
                            print(f'this is player 2, with a cosine similarity of {p2refrence} to its refrence image')
                            print(f'this is player 2, with a cosine similarity of {p2top1} to player 1 right now')
                            print(f'difference between p2 refrence and p1 right now: {abs(p2refrence-p2top1)}')
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
                        
                        
                    #putting player keypoints on the frame
                    for keypoint in kp:
                        #print(keypoint.xyn[0])
                        i=0
                        for k in keypoint.xyn[0]:
                            x, y= k
                            x=int(x*frame_width)
                            y=int(y*frame_height)
                            if playerid==1:
                                cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), 5)
                            else:
                                cv2.circle(annotated_frame, (int(x), int(y)), 3, (255, 0, 0), 5)
                            if i==16:
                                cv2.putText(
                                    annotated_frame,
                                    f"{playerid}",
                                    (int(x), int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 255, 255),
                                    3,
                                )
                            i+=1
                        
        except Exception as e:
            print("GOT ERROR: ", e)
            pass
        
        for refrence in refrence_points:
            cv2.circle(annotated_frame, (int(refrence[0]), int(refrence[1])), 5, (0, 255, 0), 2)
        
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
                # print(refrence_points)
                p1distancefromT = math.hypot(
                    refrence_points[4][0] - avgpx1, refrence_points[4][1] - avgpy1
                )
                p2distancefromT = math.hypot(
                    refrence_points[4][0] - avgpx2, refrence_points[4][1] - avgpy2
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
        combined_image = cv2.addWeighted(
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

                        #cv2.circle(
                        #    annotated_frame,
                        #    (next_pos[0], next_pos[1]),
                        #    5,
                        #    (0, 255, 0),
                        #    -1,
                        #)

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
                (int(predicted_pos[0][0]*frame_width), int(predicted_pos[0][1]*frame_height)),
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
                (int(future_predict[0][0]*frame_width), int(future_predict[0][1]*frame_height)),
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
            p1postemp=players.get(1).get_last_x_poses(3).xyn[0]
            p2postemp=players.get(2).get_last_x_poses(3).xyn[0]
            rlp1postemp = [players.get(1).get_last_x_poses(3).xyn[0][16][0]*frame_width, players.get(1).get_last_x_poses(3).xyn[0][16][1]*frame_height]
            rlp2postemp = [players.get(2).get_last_x_poses(3).xyn[0][16][0]*frame_width, players.get(2).get_last_x_poses(3).xyn[0][16][1]*frame_height]
            #print(f"Player 1: {rlp1postemp}")
            rlworldp1=Functions.pixel_to_3d(rlp1postemp, pixel_reference=refrence_points, reference_points_3d=reference_points_3d)
            rlworldp2=Functions.pixel_to_3d(rlp2postemp, pixel_reference=refrence_points, reference_points_3d=reference_points_3d)
            homop1=Functions.transform_pixel_to_real_world(rlp1postemp, homography)
            homop2=Functions.transform_pixel_to_real_world(rlp2postemp, homography)
            
            # text5=f"Player 1: {rlworldp1}"
            # text6=f"Player 2: {rlworldp2}"
            # text7=f"Player 1: {homop1}"
            # text8=f"Player 2: {homop2}"
            
            # cv2.putText(
            #     annotated_frame,
            #     text5,
            #     (10, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
            # cv2.putText(
            #     annotated_frame,
            #     text6,
            #     (10, 70),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1,
            # )
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
            # Functions.transform_and_display(rlp1postemp, rlp2postemp, pixel_reference=refrence_points, reference_points_3d=reference_points_3d, image=annotated_frame)
            
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
            '''
            with open("output/final.txt", "a") as f:
                text = f"Frame: {running_frame}{{\nPlayer 1: {p1postemp}\n\nPlayer 2: {p2postemp}\n\nBall: {mainball.getloc()}}}\n\n"
                f.write(f"{text}\n")
                f.close()
            '''
            # print(f'wrote!')
        if running_frame % 3 == 0:
            write()

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