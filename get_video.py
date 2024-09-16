from pytube import YouTube
from urllib.error import HTTPError
import logging
import os
import time

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

# Function to download video with retries
def download_video(link, output_folder, index, retries=3):
    for attempt in range(retries):
        try:
            # Log the URL being processed
            logging.info(f"Processing URL: {link}")

            # Validate the URL format
            if not link.startswith("https://www.youtube.com/watch?v="):
                raise ValueError(f"Invalid YouTube URL: {link}")

            # Download the video using pytube
            yt = YouTube(link)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

            if stream:
                stream.download(output_path=output_folder, filename=f"video_{index+1}.mp4")
                logging.info(f"Downloaded video {index+1} from {link}")
                return True  # Download successful, exit function
            else:
                logging.error(f"No suitable stream found for {link}")
                return False
        except HTTPError as e:
            logging.error(f"HTTPError for {link}: {e}")
            if attempt + 1 < retries:
                logging.info(f"Retrying ({attempt + 1}/{retries})...")
                time.sleep(2)  # Wait for a short period before retrying
            else:
                logging.error(f"Failed to download after {retries} retries: {link}")
                return False
        except ValueError as e:
            logging.error(e)
            return False
        except Exception as e:
            logging.error(f"An error occurred for {link}: {e}")
            return False

# Download and save the videos with retries
for i, link in enumerate(youtube_links):
    link = link.strip()  # Remove any leading/trailing whitespaces or newlines
    download_video(link, output_folder, i)

logging.info("All videos downloaded and saved successfully!")
