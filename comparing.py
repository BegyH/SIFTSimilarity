import os
from SIFTSimilarity import *
from tqdm import tqdm


if __name__ == "__main__":
    home_dir = os.path.join('X:/comparing/test/')
    originals = os.listdir(os.path.join(home_dir, 'orig/images/'))
    mirrors = os.listdir(os.path.join(home_dir, 'mirror/images/'))
    crops = os.listdir(os.path.join(home_dir, 'crop/images/'))
    path_for_cropped = os.path.join(home_dir, 'crop/')

    trecholds = {}

    for treshold in range(1, 17, 2):

        confusion_matrix = [[0, 0], [0, 0]]

        for orig_im in tqdm(originals):
            duplicate = 0
            first_im_path = os.path.join(home_dir, 'orig/images/', orig_im)
            uniq_counter = 0
            dupl_counter = 0

            for mir_im in mirrors:
                dup_hat = 0
                if orig_im.split('_')[0] == mir_im.split('_')[0]:
                    duplicate = 1
                else:
                    uniq_counter += 1

                if uniq_counter > 1 and duplicate == 0:
                    continue

                second_im_path = os.path.join(home_dir, 'mirror/images/', mir_im)
                similarity_score = calculate_results_for(first_im_path.replace('jpg', 'txt'),
                                                         second_im_path.replace('jpg', 'txt'),
                                                         pl=False)

                if similarity_score >= treshold:
                    dup_hat = 1

                confusion_matrix[abs(dup_hat-1)][abs(duplicate-1)] += 1

                if uniq_counter >= 1 and dupl_counter >= 1:
                    break

            # duplicate = 0
            # dupl_counter = 0

            # for crop_im in crops:
            #     dup_hat = 0
            #     if orig_im.split('_')[0] == crop_im.split('_')[0]:
            #         duplicate = 1
            #         dupl_counter += 1
            #     else:
            #         uniq_counter += 1
            #
            #     if uniq_counter > 5 and duplicate == 0:
            #         continue
            #
            #     second_im_path = os.path.join(home_dir, 'crop/images/', crop_im)
            #     similarity_score = calculate_results_for(first_im_path.replace('jpg', 'txt'),
            #                                              second_im_path.replace('jpg', 'txt'),
            #                                              pl=False)
            #
            #     if similarity_score >= treshold:
            #         dup_hat = 1
            #
            #     confusion_matrix[abs(dup_hat-1)][abs(duplicate-1)] += 1
            #
            #     if uniq_counter > 5 and dupl_counter >3:
            #         break

        if confusion_matrix[0][0] + confusion_matrix[1][0] == 0:
            recall = 1
        else:
            recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        if confusion_matrix[0][0] + confusion_matrix[0][1] == 0:
            precision = 1
        else:
            precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        f_measure = 2 * recall * precision / (recall + precision)

        trecholds[treshold] = [round(recall, 3), round(precision, 3), round(f_measure, 3)]
        trecholds['confusion_matrix'+str(treshold)] = str(confusion_matrix)

    with open(os.path.join(home_dir, 'results.txt'), 'w') as result:

        for treshold, metrics in trecholds.items():
            result.write(str(treshold) + ': ' + str(metrics) + '\n')

