import cv2
import os
import numpy as np

datasets = {
    "messidor": {
        "input": "datasets/messidor/images",
        "output": "datasets/messidor/preprocessed",
        "extensions": [".tif"]  
    },
    "diaretdb1": {
        "input": "datasets/diaretdb1/images",
        "output": "datasets/diaretdb1/preprocessed",
        "extensions": [".png"] 
    }
}

# Creating output folders
for dataset in datasets.values():
    os.makedirs(dataset["output"], exist_ok=True)

IMG_SIZE = (256, 256)

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    image = cv2.resize(image, IMG_SIZE)  

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    enhanced = cv2.equalizeHist(denoised)  

    cv2.imwrite(output_path, enhanced)

for dataset_name, paths in datasets.items():
    print(f"Processing {dataset_name} dataset...")
    for filename in os.listdir(paths["input"]):
        if any(filename.endswith(ext) for ext in paths["extensions"]):
            input_path = os.path.join(paths["input"], filename)
            output_path = os.path.join(paths["output"], filename)
            preprocess_image(input_path, output_path)

print("Image preprocessing completed! Preprocessed images are saved in respective folders.")
