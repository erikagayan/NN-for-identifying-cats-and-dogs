"""
Copies downloaded files to the dataset folder
"""

import os
import shutil

# Root folder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to source data
source_dir = os.path.join(PROJECT_DIR, "dogs-vs-cats", "train", "train")
# Path to target folders
target_base_dir = "/dataset"
cat_dir = os.path.join(target_base_dir, "cat")
dog_dir = os.path.join(target_base_dir, "dog")

# Creating Destination Folders
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# Getting a list of files
files = os.listdir(source_dir)

# Counters
moved_cats = 0
moved_dogs = 0

# File iteration and sorting
for file in files:
    if file.lower().startswith("cat"):
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(cat_dir, file)
         )
        moved_cats += 1
    elif file.lower().startswith("dog"):
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(dog_dir, file)
        )
        moved_dogs += 1

print(f"✅ Done: {moved_cats} cats and {moved_dogs} dogs moved.")
print(f"Categories are located in the following paths:\n- {cat_dir}\n- {dog_dir}")
