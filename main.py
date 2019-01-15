from detection.detect import detect
from ALPR.segmentation.plate_recognition.recognize.recognize import segmenting
import cv2


def main():
    test_img = "detection/data/5.jpg"

    img = detect(testimg=test_img, model="detection/checkpoint/model_best.pth.tar")

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    chars = segmenting(img)
    print(len(chars))





if __name__ == "__main__":
    main()
