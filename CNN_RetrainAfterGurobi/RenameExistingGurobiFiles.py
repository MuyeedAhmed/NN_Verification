import os
import re

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

root_directory = "checkpoints/"
rename_files(root_directory)