import roboflow
import os

rf = roboflow.Roboflow(api_key=os.environ["ROBOFLOWAPIKEY"])
project = rf.workspace().project("squash-black-ball")
version = project.version("1")
version.deploy("yolov11", "trained-models", "black_ball_v1(640and15epoc).pt")
