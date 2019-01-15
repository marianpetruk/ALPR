# import the necessary packages
from __future__ import print_function
from imutils import paths
import numpy as np
import imutils
import cv2

from segmentation.plate_recognition.license_plate.license_plate import LicensePlateDetector


def segmenting(image):
    # if the width is greater than 640 pixels, then resize the image
    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    # lpNumber = imagePath[-12:-4]

    # initialize the license plate detector and detect the license plates and candidates
    lpd = LicensePlateDetector(image, "AAOOOOAA")

    lp = lpd.detectCharacterCandidates()

    # candidates = np.dstack([lp.candidates] * 3)
    # thresh = np.dstack([lp.thresh] * 3)
    # output = np.vstack([lp.plate, thresh, candidates])
    # cv2.imshow("Plate & Candidates", output)
    import os
    if not os.path.exists("../../data"):
        os.makedirs("../../data")
    for i in range(len(lp.chars)):
        cv2.imshow(str(i), lp.chars[i])
        cv2.imwrite("../../data/{}.jpg".format(i), lp.chars[i])
    # display the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return lp.chars


if __name__ == "__main__":
    for imagePath in sorted(list(paths.list_images("../images/signs"))):
        img = cv2.imread(imagePath)
        print(imagePath)
        segmenting(img)



