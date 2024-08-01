import os
import json

# Load the JSON file
inference_data = 'inference'
src_folder = inference_data + '/images'
json_path = inference_data + "/full_dataset.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# Directory paths
predictions_dir = 'predictions/numpy'
new_predictions_dir = 'predictions/numpy_final'

# Ensure the new directory exists
os.makedirs(new_predictions_dir, exist_ok=True)

# Create a mapping from old filenames to new filenames
file_mapping = {}
for i, test_item in enumerate(data['test']):
    old_filename = f'plot_{i}_0.npy'
    new_filename = os.path.basename(test_item['image']).replace('.jpg', '.npy')
    file_mapping[old_filename] = new_filename

# Check if the number of old files matches the number of entries in data['test']
old_files = [f for f in os.listdir(predictions_dir) if f.endswith('.npy')]
if len(old_files) != len(data['test']):
    print(f'Warning: Number of old files ({len(old_files)}) does not match the number of test entries ({len(data["test"])})')

# Rename the files
for old_filename, new_filename in file_mapping.items():
    old_path = os.path.join(predictions_dir, old_filename)
    new_path = os.path.join(new_predictions_dir, new_filename)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        # print(f'Renamed {old_path} to {new_path}')
    else:
        print(f'File {old_path} does not exist')

print('Renaming complete.')

# Check if all new files are present in the new directory
new_files = [f for f in os.listdir(new_predictions_dir) if f.endswith('.npy')]
expected_new_files = set(file_mapping.values())
if set(new_files) != expected_new_files:
    print('Warning: The number of new files does not match the expected filenames.')
else:
    print('All files have been successfully renamed and are present in the new directory.')




import os
import shutil

def copy_json_files(src_folder, dest_folder):
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source folder
    for file_name in os.listdir(src_folder):
        # Check if file has a .json extension
        if file_name.endswith('.json'):
            # Construct full file path
            src_file = os.path.join(src_folder, file_name)
            dest_file = os.path.join(dest_folder, file_name)
            # Copy file
            shutil.copy(src_file, dest_file)
            # print(f'Copied {file_name} to {dest_folder}')

# Example usage

dest_folder = 'predictions/numpy_final'
copy_json_files(src_folder, dest_folder)
