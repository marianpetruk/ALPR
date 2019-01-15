from detection.detect import detect
import cv2


def main():
    test_img = "detection/data/1.jpg"

    img = detect(testimg=test_img, model="detection/checkpoint/model_best.pth.tar")

    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
