import os
import gdown

# Directory and file details
directory = "model_weights"
file_name = "best_PPE_detection.pt"
destination = os.path.join(directory, file_name)

# Google Drive file URL
url = f"https://drive.google.com/uc?export=download&id=1WIAupJX_HFtEoM1sONhZD6JZFsRZKzaj"

# Path for the directory
path = os.path.join(".", directory)

# Check if directory already exists
if not os.path.exists(path):
    # Create the directory
    try:
        os.makedirs(path)
        print(f"Directory '{directory}' created successfully at {path}")
    except OSError as error:
        print(f"Directory '{directory}' cannot be created due to: {error}")

# Check if the file already exists
if not os.path.exists(destination):
    # Download the file using gdown
    gdown.download(url, destination, quiet=False)
    print(f"File '{file_name}' downloaded successfully to '{destination}'")
else:
    print(f"File '{file_name}' already exists in '{destination}'")
