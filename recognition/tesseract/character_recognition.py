from os import path
import string
import json
import pytesseract
import argparse
import cv2
from tqdm import tqdm


def recognize_characters(images):
    tesseract_config = ("-l eng --oem 2 --psm 10")
    characters = []
    for img in images:
        character = pytesseract.image_to_string(img, config=tesseract_config)
        character = text_cleaning(character.lower())
        characters.append(character)

    return "".join(characters)


def text_cleaning(lower_text):
    text = set(lower_text)
    text = list(text)
    text = list(filter(lambda c: c in string.ascii_lowercase + string.digits, text))

    return "".join(text)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="path to input image")
    ap.add_argument("-east", "--east", type=str,
                    help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="nearest multiple of 32 for resized width")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="nearest multiple of 32 for resized height")
    ap.add_argument("-p", "--padding", type=float, default=0.0,
                    help="amount of padding to add to each border of ROI")
    args = vars(ap.parse_args())

    tes_config = ("-l eng --oem 2 --psm 10")
    dataset = args["image"] # SLASH MUST BE IN THE END OF DATASET PATH
    desc = dataset + "desc.json"
    with open(path.abspath(desc)) as desc_file:
        content = json.loads(desc_file.read())
        true_predictions = 0
        count = 0
        for item in tqdm(content["test"]):
            filename = item["name"]
            text_true = item["text"]
            image = cv2.imread(dataset+filename)
            orig = image.copy()
            (origH, origW) = image.shape[:2]

            # set the new width and height and then determine the ratio in change
            # for both the width and height
            (newW, newH) = (args["width"], args["height"])
            rW = origW / float(newW)
            rH = origH / float(newH)

            # resize the image and grab the new image dimensions
            image = cv2.resize(image, (newW, newH))

            text_pred = pytesseract.image_to_string(orig, config=tes_config)
            text_pred = text_cleaning(text_pred.lower())
            if text_pred.upper() == text_true.upper():
                true_predictions += 1
            count += 1
        print("Acc:", true_predictions / count)
