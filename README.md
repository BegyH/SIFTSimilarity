# SIFTSimilarity
*This scripts is working with image-algorithm SIFT and is used to check if two fotos is duplicate or is camera' angle shifted.*

Definition of file's names:
- *SIFTSimilarity.py* - file with the most important function for working with SIFT
- *CreateAnnotation.py* - I used this to make dataset images for detecting duplicates. (Calculate keypoints and descriptors for all pictures)
- *MakeAugmentations.py* - I used this to make dataset images for detecting duplicates. (Original pictures was olready exists and I made some different duplicates)
- *comparing.py* - Code for analyze dataset and calculate some metrics to understand most valuable threshold for duplicates. (in developing)
- *CheckPosition.py* - Working with camera's angles to anderstand if the prev and real-time frames is different. (in developing)
