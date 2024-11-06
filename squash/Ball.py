import math
class Ball:
    def __init__(self, x, y, a, s):
        self.xcoord=x
        self.ycoord=y
        self.pastx=0
        self.pasty=0
        self.angle=a
        self.size=s
        self.positions=[[]]
        self.positions.append([x,y])
        self.angles=[]
        if a != 0: self.angles.append(a)
        self.sizes=[]
        if s !=0: self.sizes.append(s)

    def getlastpos(self):
        return [self.pastx, self.pasty]
    def getloc(self):
        return [self.xcoord, self.ycoord]
    def get_last_x_pos(self, x):
        if len(self.positions)>x:
            return self.positions[-x]
        else:
            return None
    def update(self, x, y, s, a=0):
        self.xcoord = x
        self.ycoord = y
        self.positions.append([x, y])
        # Update other attributes...

    def getcoordhistory(self):
        return self.positions
    def getanglehistory(self):
        return self.angles
    def getsizehistory(self):
        return self.sizes

    def convert_2d_to_3d(self, court_width=6.4, court_length=9.75, camera_distance=3.5):
        """
        Converts 2D coordinates to 3D based on squash court dimensions and camera position.
        :param court_width: Width of the squash court (meters)
        :param court_length: Length of the squash court (meters)
        :param camera_distance: Distance of the camera behind the back wall (meters)
        :return: 3D coordinates [X, Y, Z]
        """
        # Normalize x and y coordinates based on court dimensions
        norm_x = self.xcoord / court_width  # X is along the width of the court
        norm_y = self.ycoord / court_length  # Y is along the length of the court

        # Map 2D coordinates into 3D court space (X, Y, Z)
        # Z is depth, which will be calculated based on the ball's relative position in the court
        X = norm_x * court_width  # X is along the width
        Y = norm_y * court_length  # Y is along the length

        # Estimate Z (depth) using the court length and camera position
        Z = camera_distance + Y  # Camera is behind the back wall, so depth decreases as Y increases

        return [X, Y, Z]
    
    def convert_to_meters(self, x,y,z,pxw, pxh):
        x=int(2*pxw/3)
        z=int(0.95*pxh)
        return [x,y,z]