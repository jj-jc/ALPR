import cv2
import numpy as np
import pytesseract as tess
tess.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

CLEAN = True

############################### Function definitions ###############################
######################################################################################

# Creation of a mask with the same size of the plate crop image, that we'll use for remove
# part of the external border of the binarized crop, in order to ease a consecutive closing
# operation
def createBorderMask(input_image, border_pixel_thickness):
    height, width = input_image.shape
    mask = input_image.copy()
    left_limit = border_pixel_thickness
    right_limit = width - border_pixel_thickness
    upper_limit = border_pixel_thickness
    lower_limit = height - border_pixel_thickness
    for row in range(0, height):
        for column in range(0, width):
            if column < left_limit or column > right_limit or row < upper_limit or row > lower_limit:
                mask[row][column] = 255
            else:
                mask[row][column] = 0
    return mask

def processImage(cv2_image):
    # -- DETECT PLATE (ROI) --

    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)                                  # Image convert to gray scale
    filtered = cv2.medianBlur(gray, 5)                                                  # Median Filter blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(filtered)  # equalize crop                                  # Adaptive equalization
    binarized = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY |
                              cv2.THRESH_OTSU)[1]                                       # Binarization using OTSU method
    kernel = np.ones((3, 3), np.uint8)                                                  # creation of Kernel for dilation
    dilated = cv2.dilate(binarized, kernel, iterations=1)
    cont, hier = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)      # Contour detection
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]                          # order contours by size, keep the bigger eight
    # look for the more "plate-like" contour
    largest_extent = 0
    for i in cont:
        x, y, w, h = cv2.boundingRect(i)                                                # Estimate closest bounding box (bb) of the contours
        aspect_ratio = w / h                                                            # calculate aspect ratio of the bb to see his "plate-likeness"
        if 7 > aspect_ratio > 1.3:                                                      # check if indeed the bounding box seems like a plate
            extent = cv2.contourArea(i)/(w*h)                                           # if so, comparate the contour with its estimated bounding box
                                                                                        # (the closest to 1 (max possible value), the more the contour looks like its bb)
            if extent > largest_extent:                                                 # the contour with bigger extent is the better candidate for being the plate, we check it here
                largest_extent = extent                                                 # We keep the contour with the biggest extent as the plate region
                crp_image = cv2_image[y:y + h, x:x + w]                                 # we crop that region from the original image

    gray_crop = cv2.cvtColor(crp_image, cv2.COLOR_BGR2GRAY)                 # convert it to grayscale

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_crop = clahe.apply(gray_crop)                                 #equalize crop

    binarized_crop = (cv2.threshold(equalized_crop, 50, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])    # Binarization using OTSU method
    filtered_crop = cv2.medianBlur(binarized_crop, 5)                       # Median filter the crop
    kernel = np.ones((3, 3), np.uint8)                                      # Kernel for closing the crop
    closed_crop = cv2.morphologyEx(filtered_crop, cv2.MORPH_CLOSE, kernel)  # Closing the crop
    kernel = np.ones((3, 3), np.uint8)                                      # Kernel for dilation the crop
    dilated_crop = cv2.dilate(closed_crop, kernel, iterations=1)            # Dilating the crop
    kernel = np.ones((3, 3), np.uint8)                                      # Kernel for erode the crop
    eroded_crop = cv2.erode(dilated_crop, kernel, iterations=1)             # Erode the crop
    mask = createBorderMask(eroded_crop, 5)                                 # Mask for reduce external border of the crop
    masked_crop = cv2.add(eroded_crop, mask)                                # Apply mask
    kernel = np.ones((3, 3), np.uint8)                                      # Kernel for closing
    closed_masked_crop = cv2.morphologyEx(masked_crop, cv2.MORPH_CLOSE,
                                          kernel)                           # Close masked crop
    plate_number = tess.image_to_string(closed_masked_crop,
                                config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 8 --oem 3')  # Apply pytesseract (OCR) and extract number plate
    return plate_number, closed_masked_crop

############################### Main program ###############################
######################################################################################
import glob
image_filenames = glob.glob('Dataset/070603/*.*')
for path in image_filenames:
    image = cv2.imread(path)
    plate_number, crp = processImage(image)
    print("License Plate is:", plate_number)
    cv2.imshow("image", image)
    cv2.imshow("crp_plate", crp)
    k = cv2.waitKey(0)
    if k == 27:                         # 'ESC' for exist
        cv2.destroyAllWindows()
        break




