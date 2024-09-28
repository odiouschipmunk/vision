from google_images_download import google_images_download

def download_images(query, num_images, save_dir, chromedriver_path):
    response = google_images_download.googleimagesdownload()
    arguments = {
        "keywords": query,
        "limit": num_images,
        "print_urls": True,
        "output_directory": save_dir,
        "image_directory": query.replace(" ", "_"),
        "chromedriver": chromedriver_path
    }
    response.download(arguments)

# Example usage
chromedriver_path = "C:\\Users\\default.DESKTOP-7FKFEEG\\chromedriver\\chromedriver-win64\\chromedriver.exe"  # Update this path
download_images("squash racket", 10000, "dataset/images/train/squash_racket", chromedriver_path)
download_images("squash ball", 10000, "dataset/images/train/squash_ball", chromedriver_path)