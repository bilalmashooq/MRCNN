import os

def update_filenames(directory):
    # Get a list of JSON files in the directory
    files = [file for file in os.listdir(directory) if file.endswith('.jpg')]

    # Sort the list of JSON files to ensure consistent numbering
    files.sort()

    # Iterate over each JSON file
    for i, file_name in enumerate(files, start=1):
        # Generate the new filename
        new_filename = f"{i:04d}.jpg"

        # Rename the file
        os.rename(os.path.join(directory, file_name), os.path.join(directory, new_filename))

        print(f"Renamed {file_name} to {new_filename}")

# Example usage
directory_path = r'C:\Users\muham\PycharmProjects\mrcnn\Val'  



from PIL import Image
import os

def compress_images(input_folder, output_folder, quality):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            # Open the image file
            img = Image.open(os.path.join(input_folder, filename))
            # Compress the image
            img.save(os.path.join(output_folder, filename), "JPEG", quality=quality)

# Specify the input and output folders
input_folder = 'Val'
output_folder = 'Val/Images'

# Call the compress_images function
compress_images(input_folder, output_folder, quality=20)


