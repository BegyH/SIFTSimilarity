from SIFTSimilarity import *
from tqdm import tqdm

if __name__ == '__main__':
    home_dir = input()
    imageList = os.listdir(os.path.join(home_dir, '/images/'))

    imagesBW = []
    for imageName in imageList:
        imagePath = home_dir + "/images/" + str(imageName)
        imagesBW.append(image_resize_train(cv2.imread(imagePath, 0)))

    sift = cv2.SIFT_create()

    # keypoints = []
    # descriptors = []

    for i, image in tqdm(enumerate(imagesBW)):
        # print("Starting for image: " + imageList[i])
        keypointTemp, descriptorTemp = compute_sift(image, sift)
        # keypoints.append(keypointTemp)
        # descriptors.append(descriptorTemp)
        # print("  Ending for image: " + imageList[i])

        deserializedKeypoints = []
        filepath = home_dir + "/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
        for point in keypointTemp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            deserializedKeypoints.append(temp)
        with open(filepath, 'wb') as fp:
            pickle.dump(deserializedKeypoints, fp)

        filepath = home_dir + "/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
        with open(filepath, 'wb') as fp:
            pickle.dump(descriptorTemp, fp)

    # for i, keypoint in enumerate(keypoints):
    #     deserializedKeypoints = []
    #     filepath = home_dir + "/keypoints/" + str(imageList[i].split('.')[0]) + ".txt"
    #     for point in keypoint:
    #         temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
    #         deserializedKeypoints.append(temp)
    #     with open(filepath, 'wb') as fp:
    #         pickle.dump(deserializedKeypoints, fp)
    #
    # for i, descriptor in enumerate(descriptors):
    #     filepath = home_dir + "/descriptors/" + str(imageList[i].split('.')[0]) + ".txt"
    #     with open(filepath, 'wb') as fp:
    #         pickle.dump(descriptor, fp)
