import cv2
import os
import cv2
import pickle
import matplotlib.pyplot as plt


def image_resize_train(image):
    maxD: int = 1024
    height, width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio), maxD)
    else:
        newSize = (maxD, int(maxD/aspectRatio))
    image = cv2.resize(image, newSize)
    return image


def image_resize_test(image):
    maxD = 1024
    height, width, channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio), maxD)
    else:
        newSize = (maxD, int(maxD/aspectRatio))
    image = cv2.resize(image, newSize)
    return image


def compute_sift(image, s):
    return s.detectAndCompute(image, None)


def fetch_keypoint_from_file(i):
    filepath = os.path.join(i)
    keypoint = []
    file = open(filepath, 'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint


def fetch_descriptor_from_file(i):
    filepath = os.path.join(i)
    file = open(filepath, 'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor


def calculate_results_for(i, j, pl: bool):

    keypoint1 = fetch_keypoint_from_file(i.replace('images', 'keypoints'))
    descriptor1 = fetch_descriptor_from_file(i.replace('images', 'descriptors'))
    keypoint2 = fetch_keypoint_from_file(j.replace('images', 'keypoints'))
    descriptor2 = fetch_descriptor_from_file(j.replace('images', 'descriptors'))
    matches = calculate_matches(descriptor1, descriptor2)
    score = calculate_score(len(matches), len(keypoint1), len(keypoint2))
    # print(len(matches), len(keypoint1), len(keypoint2), len(descriptor1), len(descriptor2))
    # print(score)
    if pl:
        plot = get_plot_for(i, j, keypoint1, keypoint2, matches)
        plt.imshow(plot), plt.show()

    return score


def get_plot_for(i, j, keypoint1, keypoint2, matches):
    image1 = image_resize_test(cv2.imread(os.path.join("data/images/", i)))
    image2 = image_resize_test(cv2.imread(os.path.join("data/images/", j)))
    return get_plot(image1, image2, keypoint1, keypoint2, matches)


def calculate_score(matches, keypoint1, keypoint2):
    return 100 * (matches/min(keypoint1, keypoint2))


def get_plot(image1, image2, keypoint1, keypoint2, matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255, 255, 255],
        flags=2
    )
    return matchPlot


def calculate_matches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    topResults1 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2, des1, k=2)
    topResults2 = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

#
# if __name__ == '__main__':
#     imageList = os.listdir(os.path.join('data/images/'))
#
#     imagesBW = []
#     for imageName in imageList:
#         imagePath = "data/images/" + str(imageName)
#         imagesBW.append(image_resize_train(cv2.imread(imagePath, 0)))
#
#     # Using opencv's sift implementation here
#     sift = cv2.SIFT_create()
#
#     keypoints = []
#     descriptors = []
#     for i, image in enumerate(imagesBW):
#         print("Starting for image: " + imageList[i])
#         keypointTemp, descriptorTemp = compute_sift(image, sift)
#         keypoints.append(keypointTemp)
#         descriptors.append(descriptorTemp)
#         print("  Ending for image: " + imageList[i])
#
#     for i, keypoint in enumerate(keypoints):
#         deserializedKeypoints = []
#         filepath = "data/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
#         for point in keypoint:
#             temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
#             deserializedKeypoints.append(temp)
#         with open(filepath, 'wb') as fp:
#             pickle.dump(deserializedKeypoints, fp)
#
#     for i, descriptor in enumerate(descriptors):
#         filepath = "data/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
#         with open(filepath, 'wb') as fp:
#             pickle.dump(descriptor, fp)


