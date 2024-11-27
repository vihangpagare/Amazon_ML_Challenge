import os

# Path to the directory containing the files
directory = "dataset/images/"

# Loop over the range of file numbers to delete
for i in range(2000, 20001):
    file_path = os.path.join(directory, f"{i}.jpg")
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except FileNotFoundError:
        print(f"File {file_path} not found")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
