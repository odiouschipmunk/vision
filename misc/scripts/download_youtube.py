import os
from pytubefix import YouTube

def download_videos(file_path, resolution='720p', output_path='.'):
    with open(file_path, 'r') as f:
        urls = f.read().splitlines()
    
    for url in urls:
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(res=resolution, progressive=True).first()
            if not stream:
                print(f"{yt.title}: {resolution} not available.")
                continue
            stream.download(output_path=output_path)
            print(f"Downloaded: {yt.title}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    download_videos('full-games.txt', '720p', './downloads')