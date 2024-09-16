import os
from pytube import YouTube
from urllib.error import HTTPError

# Path to the text file containing YouTube links
input_file = "youtube_links.txt"

# Path to the folder where the videos will be saved
output_folder = "videos"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the YouTube links from the input file
with open(input_file, "r") as file:
    youtube_links = file.readlines()

# Download and save the videos
for i, link in enumerate(youtube_links):
    # Remove any leading/trailing whitespaces or newlines
    link = link.strip()
    
    try:
        # Download the video using pytube
        yt = YouTube(link)
        yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(output_path=output_folder, filename=f"video_{i+1}.mp4")
        print(f"Downloaded video {i+1} from {link}")
    except HTTPError as e:
        print(f"HTTPError for {link}: {e}")
    except Exception as e:
        print(f"An error occurred for {link}: {e}")

print("All videos downloaded and saved successfully!")