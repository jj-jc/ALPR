import cv2
import glob
import numpy as np
import pytesseract as tess
tess.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
CLEAN = True

# function for image stacking for better visualization (no hace falta usarla, la he sacado de por ahí)
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# function for image projection calculation
def get_proj(img):
    horizontal = np.sum(img, 1)
    #horizontal = np.sum(horizontal, 1)
    vertical = np.sum(img, 0)
    #vertical = np.sum(vertical, 1)
    return horizontal, vertical


# function for image projection visualization
def plot_proj(hor_proj, ver_proj, img):
    m = np.max(hor_proj)
    w = int(500)
    result = np.zeros((hor_proj.shape[0], 500))
    for row in range(img.shape[0]):
        cv2.line(result, (0, row), (int(hor_proj[row]*w/m), row), (255, 255, 255), 1)
    cv2.imshow("Horizontal", cv2.resize(result, (500, 500)))
    w = np.max(ver_proj)
    m = int(500)
    result_ver = np.zeros((500, ver_proj.shape[0]))
    for column in range(img.shape[1]):
        cv2.line(result_ver, (column, 0), (column, int(ver_proj[column]*m/w)), (255, 255, 255), 1)
    cv2.imshow("Vertical", cv2.resize(result_ver, (500, 500)))


def image_control(j):
    k = cv2.waitKey(0)                  # wait for a key
    if k == 27:                         # 'ESC' for exist
        cv2.destroyAllWindows()
        j =- 1
    elif k == ord('n'):                  # 'n' for next
        j += 1
    elif k == ord('b'):                  # 'b' for back
        j -= 1
    if CLEAN:
        cv2.destroyAllWindows()
    return j


# Image Load into a list (Si da error fíjate bien en el path)
image_filenames = glob.glob('Dataset/070603/*.*')
image_list = []
for image in image_filenames:
    image_list.append(cv2.imread(image))


################################License plate detection###############################
######################################################################################
# Image convert to gray scale
image_gray = []
for gray in image_list:
    image_gray.append(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))

# Median Filter blur
image_filtered = []
for filtered in image_gray:
    image_filtered.append(cv2.medianBlur(filtered, 5))

# Histogram Equalization (Aquí comprueba cuantas coge bien con clahe y cuantas coge bien sin clahe, el equalizeHist
# no funciona bien porque la matrícula no representa gran parte de los píxeles
image_equalized = []
for equalized in image_filtered:
    #image_equalized.append(cv2.equalizeHist(equalized))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_equalized.append(clahe.apply(equalized))

# Binarization using OTSU method
image_binarized = []
for binarized in image_filtered:
    image_binarized.append(cv2.threshold(binarized, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])


kernel = np.ones((3, 3), np.uint8)
image_dilated = []

for dilated in image_binarized:
    image_dilated.append(cv2.dilate(dilated, kernel, iterations=1))

image_eroded = []
for eroded in image_binarized:
    image_eroded.append(cv2.erode(eroded, kernel, iterations=1))

image_closing = []
for closing in image_binarized:
    image_closing.append(cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1))

# # Edge detection
image_edge_laplacian = []
image_edge_canny = []
image_edge_sobel = []
for edge in image_dilated:
    image_edge_canny.append(cv2.Canny(edge,100, 200))
    image_edge_laplacian.append(cv2.Laplacian(edge, cv2.CV_8UC1))
    image_edge_sobel.append(cv2.Sobel(edge, cv2.CV_8UC1, 1, 1, ksize=5))

##------------------------- TO COMPARE THE PREPROCESSING ---------------##
i = 0
while i < 0:
    comparative_imag = stackImages(0.6, ([image_gray[i], image_filtered[i]], [image_equalized[i], image_binarized[i]]))
    cv2.imshow("preprocess: " + image_filenames[i], comparative_imag)
    comparative_morph = stackImages(0.6,([image_binarized[i], image_dilated[i]], [image_eroded[i], image_closing[i]]))
    cv2.imshow("morphological operations: " + image_filenames[i], comparative_morph)
    comparative_edges = stackImages(0.6, ([image_dilated[i],image_edge_canny[i]], [image_edge_laplacian[i], image_edge_sobel[i]]))
    cv2.imshow("edges: " + image_filenames[i], comparative_edges)

    i = image_control(i)
    if i == -1:
        break


# Se escoge el que mejor salga de los edge detection
image_edge = image_edge_laplacian

# Contour detection
image_contour = [[] for _ in range(len(image_list))]

index = 0
while True:
    # Primero se obtienen los contornos de la imagen y se escogen los 8 más grandes
    cont, hier = cv2.findContours(image_dilated[index], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = image_list[index].copy()
    cv2.drawContours(all_contours, cont, -1, (255, 0, 0), 3)
    #cv2.imshow("contours: "+image_filenames[index], all_contours) #Esto es para visualizar los contornos
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]
    largest_extent = 0
    compare = image_edge[index].copy()
    l = 0
    for i in cont:
        perimeter = cv2.arcLength(i, True)
        lines = cv2.approxPolyDP(i, 0.02*perimeter, True)
        x, y, w, h = cv2.boundingRect(i)
        # Ahora se hace un filtrado para descartar los contornos que no pueden ser matrícula por su aspect ratio y
        # por su extent (area del contorno/ area de la bounding box):
        aspect_ratio = w/h
        if 7 > aspect_ratio > 1.3:
            extent = cv2.contourArea(i)/(w*h)
            if extent > largest_extent:
                largest_extent = extent
                image_contour[index].clear()
                crp_image = image_list[index][y:y + h, x:x + w]
                crp_image = cv2.cvtColor(crp_image, cv2.COLOR_BGR2GRAY)
                # Esto se puede optimizar haciendo el preprocesado una vez obtenidas unicamente las imagenes de las
                # matrículas. Habría que de alguna forma eliminar esos bordes de la imagen que a veces hacen fallar al
                # OCR y además filtrar de alguna forma el escudo de Croacia
                crp_equalized = cv2.equalizeHist(crp_image)
                crp_bin_THRESHOLD = cv2.threshold(crp_equalized, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
                crp_bin = cv2.threshold(crp_equalized, crp_bin_THRESHOLD - 30, 255, cv2.THRESH_BINARY)[1]
                crp_ocr = cv2.medianBlur(crp_bin, 3)
                crp_ocr = cv2.dilate(crp_ocr, kernel, iterations=1)
                crp_ocr = cv2.erode(crp_ocr, kernel, iterations=1)
                image_contour[index].append(crp_ocr)
        cv2.rectangle(compare, (x, y), (x+w, y+h), (255, 0, 0), 2)
        l = l + 1

    cv2.imshow("original" + image_filenames[index], image_list[index])
    text = tess.image_to_string(image_contour[index][0],config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 8 --oem 3')
    print("License Plate is:", text)
    # Este codigo es para dibujar las bounding boxes de cada caracter detectado por pytesseract
    boxes = tess.image_to_boxes(image_contour[index][0], config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 8 --oem 3')
    h, w = image_contour[index][0].shape
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(image_contour[index][0], (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), 125, 2)
    cv2.imshow("cropped" + image_filenames[index], image_contour[index][0])
    index = index+1
    index = image_control(index)
    if index == -1:
        break
