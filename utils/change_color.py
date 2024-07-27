import cv2

import numpy as np
from sklearn.cluster import KMeans
import cv2
import os

def skin(color):
    temp = np.uint8([[color]])
    color = cv2.cvtColor(temp, cv2.COLOR_RGB2HSV)
    color = color[0][0]
    e8 = (color[0] <= 25) and (color[0] >= 0)
    e9 = (color[1] < 174) and (color[1] > 58)
    e10 = (color[2] <= 255) and (color[2] >= 50)
    return (e8 and e9 and e10)


# This function is meant to give the skin color of the person by detecting face and then
# apllying k-Means Clustering.
def new_skin_color(image_file):
    # Apply k-Means Clustering.
    image = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=4)
    clt.fit(image)

    def centroid_histogram(clt):
        # Grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster.
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # Normalize the histogram, such that it sums to one.
        hist = hist.astype("float")
        hist /= hist.sum()

        # Return the histogram.
        return hist

    def get_color(hist, centroids):

        # Obtain the color with maximum percentage of area covered.
        maxi = 0
        COLOR = [0, 0, 0]

        # Loop over the percentage of each cluster and the color of
        # each cluster.
        for (percent, color) in zip(hist, centroids):
            if (percent > maxi):
                if (skin(color)):
                    COLOR = color
        return COLOR

    # Obtain the color and convert it to HSV type
    hist = centroid_histogram(clt)
    skin_color = get_color(hist, clt.cluster_centers_)
    skin_temp2 = np.uint8([[skin_color]])
    skin_color = cv2.cvtColor(skin_temp2, cv2.COLOR_RGB2HSV)
    skin_color = skin_color[0][0]

    # Return the color.
    return skin_color


def segment_otsu(image_grayscale, img_BGR, vis=False):
    threshold_value, threshold_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    threshold_image_binary = 1 - threshold_image / 255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(threshold_image_binary, img_BGR)

    return img_face_only


def detect_tone(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    global_result = cv2.bitwise_not(global_mask)

    _id = np.argwhere(global_result == 0)

    b = np.mean(img[_id[:, 0], _id[:, 1]], axis=0)
    skin_temp2 = np.uint8([[b]])
    skin_color = cv2.cvtColor(skin_temp2, cv2.COLOR_BGR2HSV)
    skin_color = skin_color[0][0]

    return skin_color


def skin_tone_detect(img_BGR):
    # img_BGR = cv2.imread(image_path, 3)
    img_grayscale = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    # foreground and background segmentation (otsu)
    img_face_only = segment_otsu(img_grayscale, img_BGR)

    # convert to HSV and YCrCb color spaces and detect potential pixels
    img_face_only = img_face_only.astype(np.uint8)
    img_HSV = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2HSV)
    img_YCrCb = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2YCrCb)

    # aggregate skin pixels
    blue = []
    green = []
    red = []
    height, width, channels = img_face_only.shape
    for i in range(height):
        for j in range(width):
            if ((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (
                    90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img_face_only[i, j].item(0))
                green.append(img_face_only[i, j].item(1))
                red.append(img_face_only[i, j].item(2))
            else:
                img_face_only[i, j] = [0, 0, 0]
    skin_tone_estimate_BGR = [np.mean(blue), np.mean(green), np.mean(red)]

    skin_temp2 = np.uint8([[skin_tone_estimate_BGR]])
    skin_color = cv2.cvtColor(skin_temp2, cv2.COLOR_BGR2HSV)
    skin_color = skin_color[0][0]

    return skin_color


def skinRange(H, S, V):
    e8 = (H <= 25) and (H >= 0)
    # e9 = (S<174) and (S>58)
    e9 = (S < 174) and (S > 48)
    e10 = (V <= 255) and (V >= 50)

    return (e8 and e9 and e10)


def doDiff(img, want_color1, skin_color):
    img_BGR = img.copy()

    diff01 = want_color1[0] / skin_color[0]
    diff02 = (255 - want_color1[0]) / (255 - skin_color[0])
    diff03 = (255 * (want_color1[0] - skin_color[0])) / (255 - skin_color[0])

    diff11 = want_color1[1] / skin_color[1]
    diff12 = (255 - want_color1[1]) / (255 - skin_color[1])
    diff13 = (255 * (want_color1[1] - skin_color[1])) / (255 - skin_color[1])

    diff21 = want_color1[2] / skin_color[2]
    diff22 = (255 - want_color1[2]) / (255 - skin_color[2])
    diff23 = (255 * (want_color1[2] - skin_color[2])) / (255 - skin_color[2])

    diff1 = [diff01, diff11, diff21]
    diff2 = [diff02, diff12, diff22]
    diff3 = [diff03, diff13, diff23]

    b_index = img_BGR[:, :, 0] < skin_color[0]
    g_index = img_BGR[:, :, 1] < skin_color[1]
    r_index = img_BGR[:, :, 2] < skin_color[2]

    b_index_except = img_BGR[:, :, 0] >= skin_color[0]
    g_index_except = img_BGR[:, :, 1] >= skin_color[1]
    r_index_except = img_BGR[:, :, 2] >= skin_color[2]

    img_BGR[:, :, 0][b_index] = img_BGR[:, :, 0][b_index] * diff1[0]
    img_BGR[:, :, 1][g_index] = img_BGR[:, :, 1][g_index] * diff1[1]
    img_BGR[:, :, 2][r_index] = img_BGR[:, :, 2][r_index] * diff1[2]

    img_BGR[:, :, 0][b_index_except] = img_BGR[:, :, 0][b_index_except] * diff2[0] + diff3[0]
    img_BGR[:, :, 1][g_index_except] = img_BGR[:, :, 1][g_index_except] * diff2[1] + diff3[1]
    img_BGR[:, :, 2][r_index_except] = img_BGR[:, :, 2][r_index_except] * diff2[2] + diff3[2]
    return img_BGR


def make_lower_upper(skin_color, Hue, Saturation, Value):
    if (skin_color[0] > Hue):
        if (skin_color[0] > (180 - Hue)):
            if (skin_color[1] > Saturation + 10):
                lower1 = np.array([skin_color[0] - Hue, skin_color[1] - Saturation, Value], dtype="uint8")
                upper1 = np.array([180, 255, 255], dtype="uint8")
                lower2 = np.array([0, skin_color[1] - Saturation, Value], dtype="uint8")
                upper2 = np.array([(skin_color[0] + Hue) % 180, 255, 255], dtype="uint8")
                return (True, lower1, upper1, lower2, upper2)
            else:
                lower1 = np.array([skin_color[0] - Hue, 10, Value], dtype="uint8")
                upper1 = np.array([180, 255, 255], dtype="uint8")
                lower2 = np.array([0, 10, Value], dtype="uint8")
                upper2 = np.array([(skin_color[0] + Hue) % 180, 255, 255], dtype="uint8")
                return (True, lower1, upper1, lower2, upper2)
        else:
            if (skin_color[1] > Saturation + 10):
                lower = np.array([skin_color[0] - Hue, skin_color[1] - Saturation, Value], dtype="uint8")
                upper = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
                return (False, lower, upper)
            else:
                lower = np.array([skin_color[0] - Hue, 10, Value], dtype="uint8")
                upper = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
                return (False, lower, upper)
    else:
        if (skin_color[1] > Saturation + 10):
            lower1 = np.array([0, skin_color[1] - Saturation, Value], dtype="uint8")
            upper1 = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
            lower2 = np.array([180 - Hue + skin_color[0], skin_color[1] - Saturation, Value], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            return (True, lower1, upper1, lower2, upper2)
        else:
            lower1 = np.array([0, 10, Value], dtype="uint8")
            upper1 = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
            lower2 = np.array([180 - Hue + skin_color[0], 10, Value], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            return (True, lower1, upper1, lower2, upper2)


def change_skin(img, want_color1, skin_color, pre_skin_mask):
    # skin_color is img skin color
    # Input the image, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1 = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'

    if (skin_color[0] == 0 and skin_color[0] == 0 and skin_color[0] == 0):
        lower = np.array([0, 58, 50], dtype="uint8")
        upper = np.array([25, 173, 255], dtype="uint8")
        skinMask = cv2.inRange(converted, lower, upper)
        tmpImage = cv2.bitwise_and(img, img, mask=skinMask)
        skin_color = new_skin_color(tmpImage)
    if (skinRange(skin_color[0], skin_color[1], skin_color[2])):
        Hue = 10
        Saturation = 65
        Value = 50
        result = make_lower_upper(skin_color, Hue, Saturation, Value)
        if (result[0]):
            lower1 = result[1]
            upper1 = result[2]
            lower2 = result[3]
            upper2 = result[4]
            skinMask1 = cv2.inRange(converted, lower1, upper1)
            skinMask2 = cv2.inRange(converted, lower2, upper2)
            skinMask = cv2.bitwise_or(skinMask1, skinMask2)
        else:
            lower = result[1]
            upper = result[2]
            skinMask = cv2.inRange(converted, lower, upper)

    if pre_skin_mask is not None:
        pre_skin_mask = np.asarray(pre_skin_mask / 255, dtype=np.uint8)  # filter mask of fashion model skin out of clothes
        skinMask = pre_skin_mask * skinMask

    skinMaskInv = cv2.bitwise_not(skinMask)
    skin_color = np.uint8([[skin_color]])
    skin_color = cv2.cvtColor(skin_color, cv2.COLOR_HSV2RGB)
    skin_color = skin_color[0][0]
    skin_color = np.int16(skin_color)

    want_color1 = np.uint8([[want_color1]])
    want_color1 = cv2.cvtColor(want_color1, cv2.COLOR_HSV2RGB)
    want_color1 = want_color1[0][0]
    want_color1 = np.int16(want_color1)

    # Change the color maintaining the texture.
    img1 = doDiff(img1, want_color1, skin_color)
    img2 = np.uint8(img1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # Get the two images ie. the skin and the background.
    imgLeft = cv2.bitwise_and(img, img, mask=skinMaskInv)
    skinOver = cv2.bitwise_and(img2, img2, mask=skinMask)
    skin = cv2.add(imgLeft, skinOver)
    return skin


def merger_head_skin(input_texture, gender):

    head_hair = cv2.imread(f'app/library/data/merge_template/head_hair/head_hair_{gender}.png')
    cv_mask = cv2.imread('app/library/data/mask_face.png')
    gray_mask = cv2.cvtColor(~cv_mask, cv2.COLOR_BGR2GRAY)
    ret, thres_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

    cv_mask_eye = cv2.imread('app/library/data/mask_eye.png')
    gray_mask_eye = cv2.cvtColor(cv_mask_eye, cv2.COLOR_BGR2GRAY)
    ret_eye, thres_mask_eye = cv2.threshold(gray_mask_eye, 127, 255, cv2.THRESH_BINARY)

    user_tone = skin_tone_detect(input_texture[141:141+27, 74:74+30])
    
    skin_tone = {'female_body_skin_tone' :[10, 86, 237], 'male_body_skin_tone' : [10, 90, 208]}
    head_hair_tone = skin_tone[f"{gender}_body_skin_tone"]
    if user_tone[2] < head_hair_tone[2]:
        user_tone[2] = head_hair_tone[2]

    # head_hair_skin = change_skin(head_hair, user_tone, skin_tone[f"{gender}_body_skin_tone"], None)
    head_hair_skin = change_skin(head_hair, user_tone, head_hair_tone, None)
    mask = (thres_mask + ~thres_mask_eye)/255
    mask_compose = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    img_face = mask_compose * input_texture
    img_face = img_face.astype(np.uint8)

    monoMaskImage = cv2.split((mask_compose*255).astype(np.uint8))[0] # reducing the mask to a monochrome
    br = cv2.boundingRect(monoMaskImage) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    final_texture = cv2.seamlessClone(img_face, head_hair_skin,
                                      (mask_compose*255).astype(np.uint8), centerOfBR,
                                      cv2.NORMAL_CLONE)
    
    cv_mask_eye = cv2.imread('app/library/data/mask_eye.png')
    gray_mask_eye = cv2.cvtColor(cv_mask_eye, cv2.COLOR_BGR2GRAY)
    ret_eye, thresh_mask_eye = cv2.threshold(gray_mask_eye, 127, 255, cv2.THRESH_BINARY)
    mask_eye = np.where(thresh_mask_eye == 0)
    final_texture[mask_eye] = input_texture[mask_eye]

    return final_texture, user_tone



