import mainFunctions as mf

def main():
    player1_pos, player2_pos = mf.read_player_positions()
    player1_pos = player1_pos[0]
    player2_pos = player2_pos[0]
    reference_points_3d = mf.load_reference_points()
    #reference points in top left corner, top right corner, bottom right corner, bottom left corner, T point, left bottom of the service box, right bottom of the service box, left of tin, right of tin, left of the service line, right of the service line, left of the top line of the front court, right of the top line of the front court
    reference_points=mf.read_reference_points()
    camera_projection=mf.generate_camera_projection(reference_points, reference_points_3d)
    print(camera_projection)
if __name__ == "__main__":
    main()
