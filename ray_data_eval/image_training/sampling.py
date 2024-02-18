import os
import shutil
import random

def copy_random_folders(source_dir, target_dir, percentage=0.05, copy_dir=True):
    """
    Copies a percentage of folders from the source directory to the target directory.

    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if copy_dir:
        subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    else:
        subdirs = [d for d in os.listdir(source_dir)]

    num_to_copy = int(len(subdirs) * percentage)
    print("Number of dir: ", num_to_copy)

    # returns num_to_copy *unique* elements, so no repetition
    selected_dirs = random.sample(subdirs, num_to_copy) 

    for subdir in selected_dirs:
        src_path = os.path.join(source_dir, subdir)
        dst_path = os.path.join(target_dir, subdir)
        if copy_dir:
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

x = 10
SOURCE_DIR = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/train/"
TARGET_DIR = f"/home/ubuntu/image-data-{x}-percent/ILSVRC/Data/CLS-LOC/train/"

copy_random_folders(SOURCE_DIR, TARGET_DIR, percentage=x/100)


# make sure test, val directories contents exist so that the benchmark doesn't error, although these will not be used
source_test = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/test/"
source_val = "/home/ubuntu/image-data/ILSVRC/Data/CLS-LOC/val/"
test_dir = f"/home/ubuntu/image-data-{x}-percent/ILSVRC/Data/CLS-LOC/test/"
val_dir = f"/home/ubuntu/image-data-{x}-percent/ILSVRC/Data/CLS-LOC/val/"
copy_random_folders(source_test, test_dir, 0.01, copy_dir=False)
copy_random_folders(source_val, val_dir, 0.01, copy_dir=False)
