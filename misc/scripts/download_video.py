from pytubefix import YouTube
import os

# Define the path to the file containing YouTube links
file_path = 'misc\\scripts\\full-games.txt'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Read the YouTube links from the file
with open(file_path, 'r') as file:
    youtube_links = file.readlines()

# Strip any whitespace characters like `\n` at the end of each line
youtube_links = [link.strip() for link in youtube_links]

# Function to download a YouTube video
def download_video(url):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        print(f"Downloading {yt.title}...")
        stream.download()
        print(f"Downloaded {yt.title}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {e}")

# Download each video
for link in youtube_links:
    if link:  # Check if the link is not empty
        download_video(link)