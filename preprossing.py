from imports import *

def bluring(image):
    image = cv2.medianBlur(image, 5)
    denoised = rank.median(image, disk(3))
    return denoised

def local_hist_equal(img):
    loc = exposure.equalize_hist(img)
    return loc
