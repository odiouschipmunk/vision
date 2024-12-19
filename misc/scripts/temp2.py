from roboflow import Roboflow
import os
from tqdm import tqdm

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = "squash-vision"
projectId = "squash-black-ball"
project = rf.workspace(workspaceId).project(projectId)
for filename in tqdm(os.listdir("black_ball_frames")):
    project.upload(f"black_ball_frames/{filename}")
