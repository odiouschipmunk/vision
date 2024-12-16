import mainFunctions as mf

def main():
    player1_pos, player2_pos = mf.read_player_positions()
    mf.create_heatmap(player1_pos, player2_pos)
    player1_pos = player1_pos[0]
    player2_pos = player2_pos[0]
    reference_points_3d = mf.load_reference_points()
    #reference points in top left corner, top right corner, bottom right corner, bottom left corner, T point, left bottom of the service box, right bottom of the service box, left of tin, right of tin, left of the service line, right of the service line, left of the top line of the front court, right of the top line of the front court
    reference_points=mf.read_reference_points()
    print(reference_points)

if __name__ == "__main__":
    main()
