import os

# dirs =  ["data/input_head", "data/input_body","data/output_head","data/output_body"]
def create_directory_from_list(dirs):
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

