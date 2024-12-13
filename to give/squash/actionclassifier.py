#given player keypoints in xyn, classify it as a backhand, forehand
#player keypoints as 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

def classify(keypoints):
    # Check if keypoints are valid
    if not keypoints:
        return None
    lwrist=keypoints.xyn[0][9]
    rwrist=keypoints.xyn[0][10]
    lankle=keypoints.xyn[0][15]
    rankle=keypoints.xyn[0][16]
    lshoulder=keypoints.xyn[0][5]
    rshoulder=keypoints.xyn[0][6]
    lelbow=keypoints.xyn[0][7]
    relbow=keypoints.xyn[0][8]
    #if the wrist is above the shoulder, it is a shot
    #if the wrist is on the left of the shoulder, it is a backhand
    #if the wrist is on the right of the shoulder, it is a forehand
    