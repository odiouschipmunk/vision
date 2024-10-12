import cv2
from ultralytics import YOLO
import numpy as np
import math
# Load models
pose_model = YOLO('models/yolo11m-pose.pt')
ballmodel = YOLO('trained-models/g-ball2.pt')
#racketmodel=YOLO('trained-models/squash-racket.pt')
#courtmodel=YOLO('trained-models/court-key!.pt')
# Video file path
video_file = 'Squash Farag v Hesham - Houston Open 2022 - Final Highlights.mp4'
video_folder = 'full-games'
path = 'full-games\🇪🇬 Mickawy v Nour El Tayeb 🇪🇬  Black Ball Squash Open 2024  FREE FULL MATCH!.mp4'

cap = cv2.VideoCapture(path)
frame_width = 640  
frame_height = 360
players={}
courtref=0
occlusion_times={} 
last_frame=[]
for i in range(1, 3):
    occlusion_times[i] = 0
from Ball import Ball
# Get video dimensions
import logging
from Player import Player
max_players = 2
player_last_positions = {}
frame_count=0
trackid1=True
trackid2=True
logging.getLogger('ultralytics').setLevel(logging.ERROR) 
output_path = 'annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
fps = 25  # Frames per second
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
def sum_pixels_in_bbox(frame, bbox):
    x, y, w, h = bbox
    roi = frame[int(y):int(y+h), int(x):int(x+w)]
    return np.sum(roi)
# Create a blank canvas for heatmap based on video resolution
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
p1heatmap=np.zeros((frame_height, frame_width), dtype=np.float32)
p2heatmap=np.zeros((frame_height, frame_width), dtype=np.float32)
mainball=Ball(0,0,0,0)
ballmap=np.zeros((frame_height, frame_width), dtype=np.float32)
playerRefrence1=0
playerRefrence2=0
#other track ids necessary as since players get occluded, im just going to assign that track id to the previous id(1 or 2) to the last occluded player
#really need to fix this as if there are 2 occluded players, it will not work
otherTrackIds=[[0,0],[1,1],[2,2]]
updated=[[False, 0], [False, 0]]
def find_match_2d_array(array, x):
    for i in range(len(array)):
        if array[i][0] == x:
            return True
    return False
def findLastOne(array):
    possibleis=[]
    for i in range(len(array)):
        if array[i][1]==1:
            possibleis.append(i)
    #print(possibleis)
    if len(possibleis)>1:
        return possibleis[-1]

    return -1
def findLastTwo(array):
    possibleis=[]
    for i in range(len(array)):
        if array[i][1]==2:
            possibleis.append(i)
    if len(possibleis)>1:
        return possibleis[-1]
    return -1

def findLast(i):
    possibleits=[]
    for it in range(len(otherTrackIds)):
        if otherTrackIds[it][1]==i:
            possibleits.append(it)
    return possibleits[-1]
refrences1=[]
refrences2=[]
diff=[]
diffTrack=False
'''
def findRef(img):
    return cv2.
'''
def framepose(frame, model):
    track_results = model.track(frame, persist=True)
    try:
        if track_results and hasattr(track_results[0], 'keypoints') and track_results[0].keypoints is not None:
            global refrences1, refrences2
            # Extract boxes, track IDs, and keypoints from pose results
            boxes = track_results[0].boxes.xywh.cpu()
            track_ids = track_results[0].boxes.id.int().cpu().tolist()
            keypoints = track_results[0].keypoints.cpu().numpy()
            
            current_ids = set(track_ids)

            # Update or add players for currently visible track IDs
            #note that this only works with occluded players < 2, still working on it :(
        
            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x, y, w, h = box
                
                if not find_match_2d_array(otherTrackIds, track_id):
                    #player 1 has been updated last
                    if updated[0][1]>updated[1][1]:
                        otherTrackIds.append([track_id, 2])
                        print(f'added track id {track_id} to player 2')
                    else:
                        otherTrackIds.append([track_id, 1])
                        print(f'added track id {track_id} to player 1')
                    
                    
                '''
                not updated with otherTrackIds
                if track_ids[track_id]>2:
                    print(f'track id is greater than 2: {track_ids[track_id]}')
                    if track_ids[track_id] not in occluded_players:
                        occ_id=occluded_players.pop()

                        print(' occ id part 153 occluded player reassigned to another player that was occluded previously. this only works with <2 occluded players, fix this soon!!!!!')
                    if len(occluded_players)==1:
                        players[occluded_players.pop()]=players[track_id.get(track_id)]
                        print(' line 156 occluded player reassigned to another player that was occluded previously. this only works with <2 occluded players, fix this soon!!!!!')
                '''
                #if updated[0], then that means that player 1 was updated last
                #bc of this, we can assume that the next player is player 2
                if track_id==1:
                    playerid=1
                    refrences1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                elif track_id==2:
                    playerid=2
                    refrences2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                #updated [0] is player 1, updated [1] is player 2
                #if player1 was updated last, then player 2 is next
                #if player 2 was updated last, then player 1 is next
                #if both were updated at the same time, then player 1 is next as track ids go from 1 --> 2 im really hoping
                elif updated[0][1]>updated[1][1]:
                    playerid=2
                    #player 1 was updated last
                elif updated[0][1]<updated[1][1]:
                    playerid=1
                    #player 2 was updated last
                elif updated[0][1]==updated[1][1]:
                    playerid=1
                    #both players were updated at the same time, so we are assuming that player 1 is the next player
                else:
                    #print(f'could not find player id for track id {track_id}')
                    continue




            #player refrence appending for maybe other stuff
                if playerid==1:
                    refrences1.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    temp1=refrences1[-1]
                    #print(f'player 1 refrence: {temp1}')
                elif playerid==2:
                    refrences2.append(sum_pixels_in_bbox(frame, [x, y, w, h]))
                    temp2=refrences2[-1]
                    #print(f'player 2 refrence: {temp2}')
                



                #print(f'even though we are working with {otherTrackIds[track_id][0]}, the player id is {playerid}')
                print(otherTrackIds)
                # If player is already tracked, update their info
                if playerid in players:
                    players[playerid].add_pose(kp)
                    player_last_positions[playerid] = (x, y)  # Update position
                    players[playerid].add_pose(kp)
                    #print(f'track id: {track_id}')
                    #print(f'playerid: {playerid}')
                    if playerid==1:
                        updated[0][0]=True
                        updated[0][1]=frame_count
                    if playerid ==2:
                        updated[1][0]=True
                        updated[1][1]=frame_count
                    #print(updated)
                    # Player is no longer occluded
                    
                    #print(f"Player {playerid} updated.")
                
                # If the player is new and fewer than MAX_PLAYERS are being tracked
                if len(players) < max_players:
                    players[otherTrackIds[track_id][0]] = Player(player_id=otherTrackIds[track_id][1])
                    player_last_positions[playerid] = (x, y)
                    if playerid == 1:
                        updated[0][0]=True
                        updated[0][1]=frame_count
                    else:
                        updated[1][0]=True
                        updated[1][1]=frame_count
                    print(f"Player {playerid} added.")
    except Exception as e:
        print('GOT ERROR: ', e)
        pass

def drawmap(lx,ly,rx,ry, map):

    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), map.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), map.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), map.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), map.shape[0] - 1)
    map[ly, lx] += 1
    map[ry, rx] += 1
player_move=[[]]
courtref=np.int64(courtref)
refrenceimage=None
refmap1=np.zeros((frame_height, frame_width), dtype=np.float32)
refmap2=np.zeros((frame_height, frame_width), dtype=np.float32)
from skimage.metrics import structural_similarity as ssim_metric
def is_camera_angle_switched(frame, refrence_image, threshold=0.5):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refrence_image_gray = cv2.cvtColor(refrence_image, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_metric(refrence_image_gray, frame_gray, full=True)
    return score < threshold



import json, os
refrence_points=[]

def get_refrence_points():
    # Mouse callback function to capture click events
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            refrence_points.append((x, y))
            print(f"Point captured: ({x}, {y})")
            cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Court", frame1)

    # Function to save refrence points to a file
    def save_refrence_points(file_path):
        with open(file_path, 'w') as f:
            json.dump(refrence_points, f)
        print(f"refrence points saved to {file_path}")

    # Function to load refrence points from a file
    def load_refrence_points(file_path):
        global refrence_points
        with open(file_path, 'r') as f:
            refrence_points = json.load(f)
        print(f"refrence points loaded from {file_path}")

    # Load the frame (replace 'path_to_frame' with the actual path)
    if os.path.isfile('refrence_points.json'):
        load_refrence_points('refrence_points.json')
        print(f'Loaded refrence points: {refrence_points}')
    else:
        print('No refrence points file found. Please click on the court to set refrence points.')    
        cap2=cv2.VideoCapture(path)
        if not cap2.isOpened():
            print('Error opening video file')
            exit()
        ret1, frame1=cap2.read()
        if not ret1:
            print('Error reading video file')
            exit()
        frame1=cv2.resize(frame1, (frame_width, frame_height))
        cv2.imshow('Court', frame1)
        cv2.setMouseCallback("Court", click_event)

        print("Click on the key points of the court. Press 's' to save and 'q' to quit.\nMake sure to click in the following order shown by the example")
        example_image = cv2.imread('annotated-squashcourt.png')
        example_image_resized = cv2.resize(example_image, (frame_width, frame_height))
        cv2.imshow('Court Example', example_image_resized)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_refrence_points('refrence_points.json')
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()


get_refrence_points()

#note for anyone else seeing this:
#refrence points[10] is T, [0] is x val and [1] is y val
#refrence[0] is bottom left, 
#refrence[1] is bottom right
#refrence[2] is top right
#refrence[3] is top left
#refrence[4] is bottom middle
#refrence[5] is right bottom of square
#refrence[6] is top middle
#refrence[7] is left bottom of square
#refrence[8] is right top of square
#refrence[9] is left top of square
#refrence[10] is T
#refrence[11] is the middle of T and top middle court
btmleft=0
btmright=1
topright=2
topleft=3
btmmiddle=4
rbsq=5
topmiddle=6
lbsq=7
rtsq=8
ltsq=9
t=10
tmdl=11

theatmap1=np.zeros((frame_height, frame_width), dtype=np.float32)
theatmap2=np.zeros((frame_height, frame_width), dtype=np.float32)


# Load the reference image
reference_image = cv2.imread('court-topdown.png')
reference_image_height, reference_image_width, _ = reference_image.shape
p1xc=p1yc=p2xc=p2yc=0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    frame_count+=1
    #find a better way to get a court
    if frame_count==1:
        print('frame 1')
        courtref=np.int64(sum_pixels_in_bbox(frame, [0,0,frame_width, frame_height]))
        print(courtref)
        refrenceimage=frame
    
    #frame count for debugging
    #frame 240-300 is good for occlusion player tracking testing
    if frame_count<=200 and frame_count %1 != 0:
        continue
    if is_camera_angle_switched(frame, refrenceimage, threshold=0.6):
        print('camera angle switched')
        continue
    print(len(players))
    currentref=int(sum_pixels_in_bbox(frame, [0,0,frame_width, frame_height]))
    #general court refrence to only get the first camera angle throughout the video
    if abs(courtref - currentref) > courtref * 0.6:
        print('most likely not original camera frame')
        print('current ref: ', currentref)
        print('court ref: ', courtref)
        print(f'frame count: {frame_count}')
        print(f'difference between current ref and court ref: {abs(courtref - currentref)}')
        continue
    # Pose and ball detection
    ball = ballmodel(frame)
    pose_results = pose_model(frame)
    #racket_results=racketmodel(frame)
    #only plot the top 2 confs
    annotated_frame=pose_results[0].plot()
    #court_results=courtmodel(frame)
    # Check if keypoints exist and are not empty
    #print(pose_results)
    for refrence in refrence_points:
        cv2.circle(frame, refrence, 5, (0, 255, 0), -1)
    
    if pose_results[0].keypoints.xyn is not None and len(pose_results[0].keypoints.xyn[0]) > 0:
        for person in pose_results[0].keypoints.xyn:
            
        
            if len(person) >= 17:  # Ensure at least 17 keypoints are present

                left_ankle_x = int(person[16][0] * frame_width)  # Scale the X coordinate
                left_ankle_y = int(person[16][1] * frame_height)  # Scale the Y coordinate
                right_ankle_x = int(person[15][0] * frame_width)  # Scale the X coordinate
                right_ankle_y = int(person[15][1] * frame_height)  # Scale the Y coordinate
                if left_ankle_x > 0 or left_ankle_y > 0 or right_ankle_x > 0 or right_ankle_y > 0:
                    drawmap(left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y, heatmap)
    else:
        #print("No keypoints detected in this frame.")
        continue
    highestconf=0
    x1=x2=y1=y2=0
    # Ball detection
    #make it so that if it detects the ball in the same place multiple times it takes that out
    label=""
    for box in ball[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = ballmodel.names[int(box.cls)]
        confidence = float(box.conf)  # Convert tensor to float
        avgxtemp=int((x1temp+x2temp)/2)
        avgytemp=int((y1temp+y2temp)/2)
        '''
        if abs(avgxtemp-363)<10 and abs(avgytemp-72)<10:
            #false positive near the "V"
            #TODO find out how to check for false positives for general videos
            continue
        '''
        if confidence>highestconf:
            highestconf=confidence
            x1=x1temp
            y1=y1temp
            x2=x2temp
            y2=y2temp
    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'{label} {highestconf:.2f}', (int(x1), int(y1) - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #print(label)
    cv2.putText(annotated_frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    avg_x=int((x1+x2)/2)
    avg_y=int((y1+y2)/2)
    distance=0
    size=avg_x*avg_y
    if avg_x>0 or avg_y>0:
        if mainball.getlastpos()[0]!=avg_x or mainball.getlastpos()[1]!=avg_y:
            #print(mainball.getlastpos())
            #print(mainball.getloc())
            mainball.update(avg_x, avg_y, size)
            #print(mainball.getlastpos())
            #print(mainball.getloc())
            distance=math.hypot(avg_x-mainball.getlastpos()[0], avg_y-mainball.getlastpos()[1])
            
            with open('ball.txt', 'a') as f:
                f.write(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
                #print(f'Position(in pixels): {mainball.getloc()}\nDistance: {distance}\n')
                drawmap(mainball.getloc()[0], mainball.getloc()[1], mainball.getlastpos()[0], mainball.getlastpos()[1], ballmap)

    
    # Blur and normalize the heatmap for display
    #heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 100, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_BONE)
    ball_normalized = cv2.normalize(ballmap, None, 100, 255, cv2.NORM_MINMAX)
    ballmap_colorized = cv2.applyColorMap(ball_normalized.astype(np.uint8), cv2.COLORMAP_BONE)
    framepose(frame, pose_model)

    '''
    for box in court_results[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = courtmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #print(f'{label} {confidence:.2f} GOT COURT')
    '''
    '''
    for box in racket_results[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = racketmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f'{label} {confidence:.2f} GOT RACKET')
    '''
    # Save the heatmap

    if players.get(1) and players.get(2) is not None:
        if players.get(1).get_latest_pose() and players.get(2).get_latest_pose() is not None:
            p1x = ((players.get(1).get_latest_pose().xyn[0][16][0] + players.get(1).get_latest_pose().xyn[0][15][0]) / 2) * frame_width
            p1y = ((players.get(1).get_latest_pose().xyn[0][16][1] + players.get(1).get_latest_pose().xyn[0][15][1]) / 2) * frame_height
            p2x = ((players.get(2).get_latest_pose().xyn[0][16][0] + players.get(2).get_latest_pose().xyn[0][15][0]) / 2) * frame_width
            p2y = ((players.get(2).get_latest_pose().xyn[0][16][1] + players.get(2).get_latest_pose().xyn[0][15][1]) / 2) * frame_height
            '''
            generate heatmap!!!!

            
            p1heatmap[int(p1y), int(p1x)] += 1
            p2heatmap[int(p2y), int(p2x)] += 1

            # Check index bounds
            if 0 <= p1yc < refmap1.shape[0] and 0 <= p1xc < refmap1.shape[1]:
                refmap1[p1yc, p1xc] += 1
                print(f'updated refmap, p1: {p1xc}, {p1yc}')
            if 0 <= p2yc < refmap2.shape[0] and 0 <= p2xc < refmap2.shape[1]:
                refmap2[p2yc, p2xc] += 1
                print(f'updated refmpa, p2: {p2xc}, {p2yc}')

            refmap1_normalized = cv2.normalize(refmap1, None, 100, 255, cv2.NORM_MINMAX)
            refmap1_colored = cv2.applyColorMap(refmap1_normalized.astype(np.uint8), cv2.COLORMAP_DEEPGREEN)
            refmap2_normalized = cv2.normalize(refmap2, None, 100, 255, cv2.NORM_MINMAX)
            refmap2_colored = cv2.applyColorMap(refmap2_normalized.astype(np.uint8), cv2.COLORMAP_DEEPGREEN)
            refmap1_resized = cv2.resize(refmap1_colored, (reference_image_width, reference_image_height))
            refmap2_resized = cv2.resize(refmap2_colored, (reference_image_width, reference_image_height))

            # Blend the heatmaps with the reference image
            overlay1 = cv2.addWeighted(reference_image, 0.7, refmap1_resized, 0.3, 0)
            overlay2 = cv2.addWeighted(reference_image, 0.7, refmap2_resized, 0.3, 0)

            # Save the overlay images
            cv2.imwrite('player1_overlay.png', overlay1)
            cv2.imwrite('player2_overlay.png', overlay2)
            '''
            

    # Display ankle positions of both players
    if players.get(1) and players.get(2) is not None:
        #print('line 263')
        #print(f'players: {players}')
        #print(f'players 1: {players.get(1)}')
        #print(f'players 2: {players.get(2)}')
        #print(f'players 1 latest pose: {players.get(1).get_latest_pose()}')
        #print(f'players 2 latest pose: {players.get(2).get_latest_pose()}')
        if players.get(1).get_latest_pose() or players.get(2).get_latest_pose() is not None:
            #print('line 265')
            try:
                p1_left_ankle_x = int(players.get(1).get_latest_pose().xyn[0][16][0] * frame_width)
                p1_left_ankle_y = int(players.get(1).get_latest_pose().xyn[0][16][1] * frame_height)
                p1_right_ankle_x = int(players.get(1).get_latest_pose().xyn[0][15][0] * frame_width)
                p1_right_ankle_y = int(players.get(1).get_latest_pose().xyn[0][15][1] * frame_height)
            except Exception as e:
                p1_left_ankle_x = p1_left_ankle_y = p1_right_ankle_x = p1_right_ankle_y = 0
            try:
                p2_left_ankle_x = int(players.get(2).get_latest_pose().xyn[0][16][0] * frame_width)
                p2_left_ankle_y = int(players.get(2).get_latest_pose().xyn[0][16][1] * frame_height)
                p2_right_ankle_x = int(players.get(2).get_latest_pose().xyn[0][15][0] * frame_width)
                p2_right_ankle_y = int(players.get(2).get_latest_pose().xyn[0][15][1] * frame_height)
            except Exception as e:
                p2_left_ankle_x = p2_left_ankle_y = p2_right_ankle_x = p2_right_ankle_y = 0
            # Display the ankle positions on the bottom left of the frame
            text_p1 = f'P1 ankle positions L:({p1_left_ankle_x},{p1_left_ankle_y}) R:({p1_right_ankle_x},{p1_right_ankle_y})'
            cv2.putText(annotated_frame, f'{otherTrackIds[findLast(1)][1]}', (p1_left_ankle_x, p1_left_ankle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f'{otherTrackIds[findLast(2)][1]}', (p2_left_ankle_x, p2_left_ankle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_p2 = f'P2 ankle positions L:({p2_left_ankle_x},{p2_left_ankle_y}) R:({p2_right_ankle_x},{p2_right_ankle_y})'
            cv2.putText(annotated_frame, text_p1, (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_frame, text_p2, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            avgpx1=int((p1_left_ankle_x+p1_right_ankle_x)/2)
            avgpy1=int((p1_left_ankle_y+p1_right_ankle_y)/2)
            avgpx2=int((p2_left_ankle_x+p2_right_ankle_x)/2)
            avgpy2=int((p2_left_ankle_y+p2_right_ankle_y)/2)
            #print(refrence_points)
            text_p1t=f'P1 distance from T: {math.hypot(refrence_points[10][0]-avgpx1, refrence_points[10][1]-avgpy1)}'
            text_p2t=f'P2 distance from T: {math.hypot(refrence_points[10][0]-avgpx2, refrence_points[10][1]-avgpy2)}'
            cv2.putText(annotated_frame, text_p1t, (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated_frame, text_p2t, (10, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            #text_p1c=f'P1 coords in refrence: {p1xc}, {p1yc}'
            #text_p2c=f'P2 coords in refrence: {p2xc}, {p2yc}'
            #cv2.putText(annotated_frame, text_p1c, (10, frame_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            #cv2.putText(annotated_frame, text_p2c, (10, frame_height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    for ref in refrence_points:
        #cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(annotated_frame, (ref[0], ref[1]), 5, (0,255,0), 2)
    #cv2.imwrite('foot_placement_heatmap2.png', heatmap_colored)
    #cv2.imwrite('ball_heatmap.png', ballmap_colorized)
    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)
    '''
    COURT DETECTION
    for box in court[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = courtmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

'''
    out.write(annotated_frame)
    '''
    print(f'frame count: {frame_count}')
    print(f'refrences1: {refrences1}')
    print(f'refrences2: {refrences2}')
    print(f'avg refrences1: {sum(refrences1)/len(refrences1)}')
    print(f'avg refrences2: {sum(refrences2)/len(refrences2)}')
    if len(refrences1)>0 and len(refrences2)>0:
        print(f'avg differences:{(sum(refrences1)/len(refrences1))-(sum(refrences2)/len(refrences2))}')
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()