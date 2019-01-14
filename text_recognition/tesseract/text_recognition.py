# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages
from os import path
import string
import json
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


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

# # load the input image and grab the image dimensions
# image = cv2.imread(args["image"])
# orig = image.copy()
# (origH, origW) = image.shape[:2]
#
# # set the new width and height and then determine the ratio in change
# # for both the width and height
# (newW, newH) = (args["width"], args["height"])
# rW = origW / float(newW)
# rH = origH / float(newH)
#
# # resize the image and grab the new image dimensions
# image = cv2.resize(image, (newW, newH))
# (H, W) = image.shape[:2]
#
# # define the two output layer names for the EAST detector model that
# # we are interested -- the first is the output probabilities and the
# # second can be used to derive the bounding box coordinates of text
# layerNames = [
#     "feature_fusion/Conv_7/Sigmoid",
#     "feature_fusion/concat_3"]
#
# # load the pre-trained EAST text detector
# print("[INFO] loading EAST text detector...")
# net = cv2.dnn.readNet(args["east"])
#
# # construct a blob from the image and then perform a forward pass of
# # the model to obtain the two output layer sets
# blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
#                              (123.68, 116.78, 103.94), swapRB=True, crop=False)
# net.setInput(blob)
# (scores, geometry) = net.forward(layerNames)
#
# # decode the predictions, then  apply non-maxima suppression to
# # suppress weak, overlapping bounding boxes
# (rects, confidences) = decode_predictions(scores, geometry, args["min_confidence"])
# boxes = non_max_suppression(np.array(rects), probs=confidences)
#
# # initialize the list of results
# results = []
#
# # loop over the bounding boxes
# for (startX, startY, endX, endY) in boxes:
#     # scale the bounding box coordinates based on the respective
#     # ratios
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)
#
#     # in order to obtain a better OCR of the text we can potentially
#     # apply a bit of padding surrounding the bounding box -- here we
#     # are computing the deltas in both the x and y directions
#     dX = int((endX - startX) * args["padding"])
#     dY = int((endY - startY) * args["padding"])
#
#     # apply padding to each side of the bounding box, respectively
#     startX = max(0, startX - dX)
#     startY = max(0, startY - dY)
#     endX = min(origW, endX + (dX * 2))
#     endY = min(origH, endY + (dY * 2))
#
#     # extract the actual padded ROI
#     roi = orig[startY:endY, startX:endX]
#
#     # in order to apply Tesseract v4 to OCR text we must supply
#     # (1) a language, (2) an OEM flag of 4, indicating that the we
#     # wish to use the LSTM neural net model for OCR, and finally
#     # (3) an OEM value, in this case, 7 which implies that we are
#     # treating the ROI as a single line of text
#     config = ("-l eng --oem 2 --psm 10")
#     text = pytesseract.image_to_string(roi, config=config)
#
#     # add the bounding box coordinates and OCR'd text to the list
#     # of results
#     results.append(((startX, startY, endX, endY), text))
#
# # sort the results bounding box coordinates from top to bottom
# results = sorted(results, key=lambda r: r[0][1])
#
# # loop over the results
# for ((startX, startY, endX, endY), text) in results:
#     # display the text OCR'd by Tesseract
#     print("OCR TEXT")
#     print("========")
#     print("{}\n".format(text))
#
#     # strip out non-ASCII text so we can draw the text on the image
#     # using OpenCV, then draw the text and a bounding box surrounding
#     # the text region of the input image
#     text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
#     output = orig.copy()
#     cv2.rectangle(output, (startX, startY), (endX, endY),
#                   (0, 0, 255), 2)
#     cv2.putText(output, text, (startX, startY - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#
#     # show the output image
#     plt.imshow(output)
#     plt.show()
#
# config = ("-l eng --oem 2 --psm 10")
# text = pytesseract.image_to_string(orig, config=config)
# print("OCR TEXT")
# print("========")
# print("{}\n".format(text))
# plt.imshow(orig)
# plt.show()

def text_cleaning(lower_text):
    text = set(lower_text)
    text = list(text)
    text = list(filter(lambda c: c in string.ascii_lowercase + string.digits, text))
    return "".join(text)

if __name__ == "__main__":
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
            # print("TRUE TEXT")
            # print("========")
            # print("{}\n".format(text_true.upper()))
            # print("OCR TEXT")
            # print("========")
            # print("{}\n".format(text_pred.upper()))
            #plt.imshow(orig)
            #plt.show()
        print("Acc:", true_predictions / count)
