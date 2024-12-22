import mainFunctions as mf


def main():
    player1_pos, player2_pos = mf.read_player_positions()
    # reference points in top left corner, top right corner, bottom right corner, bottom left corner, T point, left bottom of the service box, right bottom of the service box, left of tin, right of tin, left of the service line, right of the service line, left of the top line of the front court, right of the top line of the front court
    mf.read_ball_positions()
    reference_points = mf.read_reference_points()
    reference_points_3d = mf.load_reference_points()
    rlplayer1pos, rlplayer2pos, rlballpos = mf.read_rl_positions()
    print(rlballpos)
    mf.visualize_3d_ball_position(reference_points_3d, rlballpos)
    #mf.visualize_3d_ball_position(reference_points_3d, rlballpos)


if __name__ == "__main__":
    main()
