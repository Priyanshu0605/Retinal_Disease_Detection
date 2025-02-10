import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

diaretdb1_images_path = "datasets/diaretdb1/preprocessed"
diaretdb1_annotations_path = "datasets/diaretdb1/groundtruth"

lesion_types = ["Haemorrhages", "Hard_exudates", "Red_small_dots"]
num_classes = len(lesion_types) 

def get_lesion_labels():
    labels = {}
    
    for filename in os.listdir(diaretdb1_annotations_path):
        if filename.endswith(".xml") and "_plain" not in filename:
            file_path = os.path.join(diaretdb1_annotations_path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            base_name = "_".join(filename.split("_")[:2]) + ".png"  # base image filename (without _XX.xml)
            if base_name not in labels:
                labels[base_name] = [0] * num_classes

            for marking in root.findall(".//marking"):
                marking_type = marking.find("markingtype").text.strip()
                confidence = marking.find("confidencelevel").text.strip() 
                if marking_type in lesion_types and confidence in ["Medium", "High"]:
                    labels[base_name][lesion_types.index(marking_type)] = 1 

    return labels

lesion_labels = get_lesion_labels()
X, y = [], []

for filename in os.listdir(diaretdb1_images_path):
    if filename in lesion_labels:
        img_path = os.path.join(diaretdb1_images_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
        X.append(img)
        y.append(lesion_labels[filename])

X = np.array(X) / 255.0  # Normalization
y = np.array(y)  

X = np.expand_dims(X, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("X_lesion_train.npy", X_train)
np.save("X_lesion_test.npy", X_test)
np.save("y_lesion_train.npy", y_train)
np.save("y_lesion_test.npy", y_test)

print("Lesion dataset prepared successfully!")