import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f