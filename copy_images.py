import os


# Set the paths
nyu_data_path = "nyu_data/data/nyu2_train"
images_folder = "images"
masks_folder = "masks"

folder_counter = 0

# Iterate through folders in nyu2_train
for folder_name in os.listdir(nyu_data_path):
    if folder_counter >= 10:
        break

    folder_path = os.path.join(nyu_data_path, folder_name)

    # Iterate through files in the directory
    for image_name in (os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)


        # Copy the file to the appropriate folder with the unique ID
        if image_name.endswith(".jpg"):
            with open(image_path, 'rb') as src, open(os.path.join(images_folder, f"{folder_name}_{image_name}"), 'wb') as dest:
                dest.write(src.read())
        elif image_name.endswith(".png"):
            with open(image_path, 'rb') as src, open(os.path.join(masks_folder, f"{folder_name}_{image_name}"), 'wb') as dest:
                dest.write(src.read())
    folder_counter += 1

print("Files copied successfully")