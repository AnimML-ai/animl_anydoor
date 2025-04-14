from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm
import json

# Load dataset
# dataset = load_dataset("SaffalPoosh/VITON-HD-test", split="train")
dataset = load_dataset("forgeml/viton_hd", split="train")

# Output root directory
# output_root = "viton_hd_by_feature"
output_root = "viton_hd"
os.makedirs(output_root, exist_ok=True)

# Loop over each feature (column) in the dataset
for feature in dataset.features:
    # Only save image-like features
    if dataset[0][feature] is None:
        continue

    if isinstance(dataset[0][feature], Image.Image):
        print(f"{feature} = Image")
    elif isinstance(dataset[0][feature], str):
        print(f"{feature} = String")
    else:
        print("Class unknowd")

    feature_folder = os.path.join(output_root, feature)
    os.makedirs(feature_folder, exist_ok=True)

    print(f"Processing {feature}")

captions = {}
for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="dataset"):
    for feature in sample.keys():
        feature_folder = os.path.join(output_root, feature)
        if isinstance(sample[feature], Image.Image):
            dst_image_path = os.path.join(feature_folder, f"{feature}_{i:05}.jpg")
            if not os.path.exists(dst_image_path):
                img = sample[feature]
                img.convert("RGB").save(dst_image_path)
        elif isinstance(sample[feature], str):
            captions[f"{feature}_{i:05}"] = sample[feature]

with open(f"{output_root}/captions.json", "w") as f:
    json.dump(captions, f)
