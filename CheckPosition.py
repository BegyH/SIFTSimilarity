import SIFTSimilarity
import cv2
import os
import numpy as np


def cut_images(first_im, second_im) -> dict:
    """
    function creates two directories with cropping edges of two images.
    Using to comparing corresponding patrs of images.
    :param first_im: path to the original camera' angle
    :param second_im: path to the verifiable camera' angle
    :return: dict containing 6 parts of both images with labels in keys.
    """

    sizes = [[first_im.shape[0], second_im.shape[0]],
             [first_im.shape[1], second_im.shape[1]]]

    # cropping images and take them labels
    corner_pieces = {'left_top': [first_im[0:int(sizes[0][0] * 0.1),
                                           0:int(sizes[1][0] * 0.1)],
                                  second_im[0:int(sizes[0][1] * 0.1),
                                            0:int(sizes[1][1] * 0.1)]],
                     'left_middle': [first_im[int(sizes[0][0] * 0.45):int(sizes[0][0] * 0.55),
                                              0:int(sizes[1][0] * 0.1)],
                                     second_im[int(sizes[0][1] * 0.45):int(sizes[0][1] * 0.55),
                                               0:int(sizes[1][1] * 0.1)]],
                     'left_bottom': [first_im[int(sizes[0][0] * 0.9):sizes[0][0],
                                              0:int(sizes[1][0] * 0.1)],
                                     second_im[int(sizes[0][1] * 0.9):sizes[0][1],
                                               0:int(sizes[1][1] * 0.1)]],
                     'right_top': [first_im[0:int(sizes[0][0] * 0.1),
                                            int(sizes[1][0] * 0.9):sizes[1][0]],
                                   second_im[0:int(sizes[0][1] * 0.1), int(sizes[1][1] * 0.9):sizes[1][1]]],
                     'right_middle': [first_im[int(sizes[0][0] * 0.45):int(sizes[0][0] * 0.55),
                                      int(sizes[1][0] * 0.9):sizes[1][0]],
                                      second_im[int(sizes[0][1] * 0.45):int(sizes[0][1] * 0.55),
                                      int(sizes[1][1] * 0.9):sizes[1][1]]],
                     'right_bottom': [first_im[int(sizes[0][0] * 0.9):sizes[0][0], int(sizes[1][0] * 0.9):sizes[1][0]],
                                      second_im[int(sizes[0][0] * 0.9):sizes[0][1],
                                      int(sizes[1][1] * 0.9):sizes[1][1]]]}

    return corner_pieces


def compare_pieces(orig, checked, s) -> dict:
    """

    :param s: sift example.
    :param orig: pieces from original image.
    :param checked: pieces from check-angle image.
    :return: tuple of keypoints and descriptors of each images.
    """

    orig_keyp, orig_desc = SIFTSimilarity.compute_sift(orig, s)
    check_keyp, check_desc = SIFTSimilarity.compute_sift(checked, s)
    matches = SIFTSimilarity.calculate_matches(orig_desc, check_desc)
    score = SIFTSimilarity.calculate_score(matches, orig_keyp, check_keyp)
    ans = {'score': score,
           'keypoints': [orig_keyp, check_keyp],
           'matches': matches}
    return ans


def compare_images(_fist, _second) -> bool:
    """

    :param _fist: original image with truly good position.
    :param _second: suspicious-angle' camera angle that must be checked.
    :return: True - if camera is shifted and Bool in other case.
    """

    sift = cv2.SIFT_create()
    corners = cut_images(_fist, _second)
    sum_score = 0
    for label in corners:
        stats = compare_pieces(corners[label][0], corners[label][1], sift)
        sum_score += stats['score']
    mean_score = sum_score / 6

    return mean_score < 70


if __name__ == "__main__":
    ans = compare_images(cv2.imread('C:/Users/User/Desktop/test/1.png'), cv2.imread('C:/Users/User/Desktop/test/2.png'))
    print(ans)
