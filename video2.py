#from pytube import YouTube
from urllib.error import HTTPError
import logging
import os
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the text file containing YouTube links
input_file = "youtube_links.txt"

# Path to the folder where the videos will be saved
output_folder = "videos"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the YouTube links from the input file
with open(input_file, "r") as file:
    youtube_links = file.readlines()


for i, link in enumerate(youtube_links):
    link=link.strip()
    yt=YouTube(link, on_progress_callback=on_progress)
    print(yt.title)
    ys=yt.streams.get_highest_resolution()
    ys.download('videos')
logging.info("All videos downloaded and saved successfully!")
