import cv2
from PIL import Image
from squash import Functions
from squash.Player import Player
def framepose(pose_model, frame, otherTrackIds, updated, references1, references2, pixdiffs, players, frame_count, player_last_positions, frame_width, frame_height, annotated_frame, max_players=2):
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
            player_image = Image.fromarray(player_crop)
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
                            0.8,
                            (255, 255, 255),
                            3,
                        )
                    i += 1
    return [pose_model, frame, otherTrackIds, updated, references1, references2, pixdiffs, players, frame_count, player_last_positions, frame_width, frame_height, annotated_frame]