from detection.detect import detect
from segmentation.plate_recognition.recognize.recognize import segmenting
import cv2


def main():
    test_img = "detection/data/1.jpg"

    img = detect(testimg=test_img, model="detection/checkpoint/model_best.pth.tar")

    chars = segmenting(img)


    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
