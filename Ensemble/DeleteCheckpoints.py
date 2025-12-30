import os
import re
import shutil

def delete_checkpoints(root_dir):

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "GE_RAF" not in filename:
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

            
def rename_files(root_dir):
    pattern = re.compile(r'^(.*)_GurobiEdit(\.[^./\\]+)$')

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                old_path = os.path.join(dirpath, filename)
                new_filename = f"{match.group(1)}_GE_RAB{match.group(2)}"
                new_path = os.path.join(dirpath, new_filename)
                
                print(f"Renaming: {old_path} -> {new_path}")
                os.rename(old_path, new_path)


delete_checkpoints("G_checkpoints/checkpoints_A1")

# delete_checkpoints("checkpoints/")

