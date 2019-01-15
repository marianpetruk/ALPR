from detection.detect import detect
from ALPR.segmentation.plate_recognition.recognize.recognize import segmenting
from ALPR.recognition.knn.character_recognition import recognize_characters as knn_recognize
from ALPR.recognition.tesseract.character_recognition import recognize_characters as tesseract_recognize

import cv2


def main():
    test_img = "detection/data/5.jpg"

    img = detect(testimg=test_img, model="detection/checkpoint/model_best.pth.tar")

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    chars = segmenting(img)
    print(len(chars))

    knn_predicted_plate_number = knn_recognize(chars)
    print("knn predict:", knn_predicted_plate_number)
    img_copy = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, knn_predicted_plate_number, (10, 500), font, 4, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imshow("KNN", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    tesseract_predicted_plate_number = tesseract_recognize(chars)
    print("tesseract predict:", tesseract_predicted_plate_number)

    img_copy = img.copy()
    cv2.putText(img, tesseract_predicted_plate_number, (10, 500), font, 4, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imshow("Tesseract", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
