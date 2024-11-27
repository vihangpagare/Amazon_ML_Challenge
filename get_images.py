import os
from PIL import Image
import requests
import pandas as pd

df = pd.read_csv("dataset/test.csv")

# Function to download and save images
def download_and_save_images(image_urls, save_dir):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, url in enumerate(image_urls, start=1):
        try:
            # Download the image
            image = Image.open(requests.get(url, stream=True).raw)
            
            # Define the file path with increasing order filenames
            file_path = os.path.join(save_dir, f"{idx}.jpg")
            
            # Save the image
            image.save(file_path)
            print(f"Saved: {file_path}")
        
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Example usage
save_directory = "dataset/test_images"
image_urls = df['image_link'].tolist()
download_and_save_images(image_urls, save_directory)
