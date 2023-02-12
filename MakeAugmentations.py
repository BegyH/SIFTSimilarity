import os
import albumentations as alb
from albumentations.augmentations import crop as crp
import cv2
from random import randint
import numpy as np
from tqdm import tqdm


def make_crop_duplicates(path: str, copies: int) -> int:
    img = cv2.imread(path)
    for i in range(copies):
        walls = {0: (0.001, 1),
                 1: (0.001, randint(60, 100) * 0.01),
                 2: (randint(1, 40) * 0.01, 1)}
        width_border = randint(0, 2)
        height_border = randint(0, 2)
        width = (walls[width_border][0] * img.shape[0], walls[width_border][1] * img.shape[0])
        height = (walls[height_border][0]*img.shape[1], walls[height_border][1]*img.shape[1])
        crop = img[int(width[0]): int(width[1]), int(height[0]): int(height[1])]
        crop_path = path.replace('.', f'_{i}.').replace('orig', 'crop')
        cv2.imwrite(crop_path, crop)

    return 0


def make_mirror_duplicate(path: str) -> int:
    img = cv2.imread(path)
    B = img[:, :, 0].T
    G = img[:, :, 1].T
    R = img[:, :, 2].T
    img = img.T
    mirror_B = []
    mirror_G = []
    mirror_R = []
    for col in range(img.shape[1] - 1, -1, -1):
        mirror_B.append(B[col, :])
        mirror_R.append(R[col, :])
        mirror_G.append(G[col, :])
    mirror = np.concatenate([np.array([mirror_B]).T, np.array([mirror_G]).T, np.array([mirror_R]).T], axis=2)
    mir_path = path.replace('.', '_.').replace('orig', 'mirror')
    cv2.imwrite(mir_path, mirror)

    return 0


if __name__ == "__main__":
    home_dir = os.path.join('X:/comparing/test/')
    originals = os.listdir(os.path.join(home_dir, 'orig/images/'))
    path_for_mirrored = os.path.join(home_dir, 'mirror/images/')
    path_for_cropped = os.path.join(home_dir, 'crop/images/')

    for im in tqdm(originals):
        path = os.path.join(home_dir, 'orig/images/', im)
        # make_crop_duplicates(path, 3)
        make_mirror_duplicate(path)
