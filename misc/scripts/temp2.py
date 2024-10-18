from PIL import Image

# Create a white image of size 640x360
width, height = 640, 360
white_image = Image.new('RGB', (width, height), color='white')

# Save the image as white.png
white_image.save('white.png')