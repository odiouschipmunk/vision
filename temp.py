from roboflow import Roboflow

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="8kkmyg5axTR1VHCPGKlf")

# Retrieve your current workspace and project name
print(rf.workspace())
import os
# Specify the project for upload
# let's you have a project at https://app.roboflow.com/my-workspace/my-project
workspaceId = 'squash-2ezxm'
projectId = 'vision-qbw66'
project = rf.workspace(workspaceId).project(projectId)

# Upload the image to your project

"""
Optional Parameters:
- num_retry_uploads: Number of retries for uploading the image in case of failure.
- batch_name: Upload the image to a specific batch.
- split: Upload the image to a specific split.
- tag: Store metadata as a tag on the image.
- sequence_number: [Optional] If you want to keep the order of your images in the dataset, pass sequence_number and sequence_size..
- sequence_size: [Optional] The total number of images in the sequence. Defaults to 100,000 if not set.
"""
def uploadframes(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        project.upload(
            image_path=path,
            batch_name="frames",
            split="train",
            num_retry_uploads=3,
        )
uploadframes("frames")