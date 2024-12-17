import mainFunctions as mf

def main():
    player1_pos, player2_pos = mf.read_player_positions()
    ball_positions = mf.read_ball_positions()
    reference_points_3d = mf.load_reference_points()
    #reference points in top left corner, top right corner, bottom right corner, bottom left corner, T point, left bottom of the service box, right bottom of the service box, left of tin, right of tin, left of the service line, right of the service line, left of the top line of the front court, right of the top line of the front court
    rlplayer1pos, rlplayer2pos, rlballpos = mf.read_rl_positions()
    print(rlplayer1pos)
    print(rlplayer2pos)
    print(rlballpos)
    mf.visualize_3d_animation(reference_points_3d, rlplayer1pos, rlplayer2pos)
if __name__ == "__main__":
    main()