import os
import random
import shutil
from glob import glob

ROOT = r"D:\Hieu\B3\Group Project\Dataset\TUCHUp"

IMG_DIR = os.path.join(ROOT, "images")
LBL_DIR = os.path.join(ROOT, "labels")

TRAIN_RATIO = 0.8  # 80% train, 20% val


IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.HEIC"]


def collect_images(img_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(img_dir, ext)))
    return files


def remove_unlabeled_images(images, lbl_dir):
   
    cleaned_images = []
    removed_count = 0

    for img_path in images:
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)
        lbl_name = stem + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if os.path.exists(lbl_path):
            cleaned_images.append(img_path)
        else:
           
            try:
                os.remove(img_path)
                removed_count += 1
                print(f"Image without labels: {img_path}")
            except Exception as e:
                print(f"Failed to delete image {img_path}: {e}")

    print(f"Deleted {removed_count} images without labels.")
    print(f"Remaining {len(cleaned_images)} images with full labels.")
    return cleaned_images


def main():
    # 1. get all images
    images = collect_images(IMG_DIR)
    if not images:
        raise RuntimeError(f"Not found images in {IMG_DIR}")

    print("Total initial images:", len(images))

    # 1.5. delete images without labels
    images = remove_unlabeled_images(images, LBL_DIR)
    if not images:
        raise RuntimeError("All images are missing labels, cannot split dataset.")

    # 2. Shuffle & split train/val
    random.shuffle(images)
    n_train = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    print(f"Number of train images: {len(train_imgs)}")
    print(f"Number of val images:   {len(val_imgs)}")

    # 3. Make dirs
    train_img_dir = os.path.join(ROOT, "images", "train")
    val_img_dir   = os.path.join(ROOT, "images", "val")
    train_lbl_dir = os.path.join(ROOT, "labels", "train")
    val_lbl_dir   = os.path.join(ROOT, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir,   exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir,   exist_ok=True)

    # 4. Function to move image-label pair
    def move_pair(img_path, dst_img_dir, dst_lbl_dir):
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)
        lbl_name = stem + ".txt"
        lbl_path = os.path.join(LBL_DIR, lbl_name)

        # copy/move image
        shutil.copy2(img_path, os.path.join(dst_img_dir, fname))

        # copy/move label if exists
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, lbl_name))
        else:
            # could log warning here if needed
            print(f"Not found label for: {fname}")

    # 5. Copy train
    print("Starting to split train...")
    for img in train_imgs:
        move_pair(img, train_img_dir, train_lbl_dir)

    # 6. Copy val
    print("Starting to split val...")
    for img in val_imgs:
        move_pair(img, val_img_dir, val_lbl_dir)

    print("Completed splitting train/val!")

if __name__ == "__main__":
    main()
