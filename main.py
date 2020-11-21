
import cv2
import glob
import numpy as np


CLEAN=True

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
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
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
def getProj(img):
    horizontal = np.sum(img, 1)
    vertical = np.sum(np.transpose(img), 1)
    return horizontal, vertical
# function for image projection visualization
def plotProj(hor_proj, ver_proj, img):
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


def image_control(i):
    k = cv2.waitKey(0)                  # wait for a key 
    if k == 27:                         # 'ESC' for exist
        cv2.destroyAllWindows() 
        i=-1
    elif k == ord('n'):                  # 'n' for next
        i+=1        
    elif k == ord('b'):                  # 'b' for back
        i-=1
    if (CLEAN == True):
        cv2.destroyAllWindows()
    return i

# Image Load into a list (Si da error fíjate bien en el path)
image_filenames = glob.glob('Dataset/070603/*.*')
image_list = []
for image in image_filenames:
    image_list.append(cv2.imread(image))




################################License plate detection################################
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
    image_eroded.append(cv2.erode(eroded, kernel, iterations=4))

image_closing = []
for closing in image_binarized:
    image_closing.append(cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=4))


# # Laplacian Edge Detection, Canny can be used too (Si quieres, puedes intentar hacer funcionar el de Sobel bien
# # con otros parámetros, a mi me sale horrible)
image_edge_laplacian = []
image_edge_canny=[]
image_edge_sobel=[]
#filter = np.array([-1, 0, 1],[-2, 0, 2],[-1, 0, 1])
for edge in image_dilated:
    image_edge_canny.append(cv2.Canny(edge,100, 200))
    image_edge_laplacian.append(cv2.Laplacian(edge, cv2.CV_8UC1))
    image_edge_sobel.append(cv2.Sobel(edge, cv2.CV_8UC1, 1, 1, ksize=5))

##------------------------- TO COMPARE THE PREPROCESSING ---------------##
i=0;
while True:
    comparative_imag = stackImages(0.6,([image_gray[i],image_filtered[i]],[image_equalized[i],image_binarized[i]]))
    cv2.imshow("preprocess: "+ image_filenames[i],comparative_imag)
    comparative_morph = stackImages(0.6,([image_binarized[i],image_dilated[i]],[image_eroded[i],image_closing[i]]))
    cv2.imshow("morphological operations: "+ image_filenames[i],comparative_morph)
    comparative_edges = stackImages(0.6,([image_dilated[i],image_edge_canny[i]],[image_edge_laplacian[i],image_edge_sobel[i]]))
    cv2.imshow("edges: "+ image_filenames[i],comparative_edges)

    i=image_control(i)
    if( i==-1):
        break;


# Se escoge el que mejor salga de los edge detection
image_edge=image_edge_laplacian



# # Edge dilation (Realmente no está puesta ninguna dilation porque el kernel es (1, 1), puedes probar si con (2, 2) va
# # mejor, desde luego más de (2, 2) es una burrada), (el erode empeora más que mejorar, y el resto de operaciones
# # morfologicas no parecen mejorar tampoco (open, close, tophat, blackhat, etc))
# image_dilated = []
# # image_eroded = []
# dilate_kernel = np.ones((1, 1), np.uint8)
# erode_kernel = np.ones((2, 2), np.uint8)
# for dilate in image_edge:
#     image_dilated.append(cv2.dilate(dilate, dilate_kernel, iterations=1))
#     #image_dilated.append(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
#     a = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     #cv2.imshow("LUL", cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
#     #cv2.waitKey(0)
# # for erode in image_dilated:
# #     image_eroded.append(cv2.erode(erode, erode_kernel, iterations=1))

# Contour detection
image_contour = []

index=0;
while True:
    compares=[]
    #original = image_edge[index] 
    original = image_dilated[index] # crea una copia de la imagen preprocesada para trabajar sobre ella
    cont, hier = cv2.findContours(original, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # encuentra los
    # contornos de la imagen y los almacena en un vector "cont". La variable "hier" no la usamos porque tiene que ver
    # con la jerarquía que toman los contornos entre ellos.
    all_contours = image_list[index].copy()
    cv2.drawContours(all_contours, cont, -1, (255, 0, 0), 3) # estas dos líneas junto con la siguiente que está
    # comentada muestran los contornos detectados sobre la imagen original para visualización
    cv2.imshow("contours: "+image_filenames[index], all_contours)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8] # esto sirve para obtener los 8 contornos con más área
    # y así no tener en cuenta contornos muy pequeños que puede haber en la imagen. A partir de estos 8 contornos o los
    # que se se desee se realiza la búsqueda de la matrícula

    compare=image_edge[index].copy()

    for i in cont:
        perimeter = cv2.arcLength(i, True)
        lines = cv2.approxPolyDP(i, 0.02*perimeter, True) # estas dos funciones aproximan cada contorno en un
        # polinomio de "n" líneas
        # if len(lines) == 4: # se estudian los contornos de 4 lineas (cuadrados/rectángulos), que es lo que nos
        # interesa ya que las matrículas de coche son rectángulos
        x, y, w, h = cv2.boundingRect(i) # aquí se obtiene un rectángulo que contiene al contorno estudiado
        crp_image = original[y:y+h, x:x+w] # se recorta la imagen para solo tener el contorno estudiado
        hor_proj, ver_proj = getProj(crp_image) # aquí se obtienen y se grafican las proyecciones vertical y
        #plotProj(hor_proj, ver_proj, crp_image) # horizontal (igual no sirven para nada o igual las podemos usar)

        cv2.rectangle(compare, (x, y), (x+w, y+h), (255, 0, 0), 2) # Im
        #breakagen auxiliar "compare" sobre la cual se
        # pinta el rectángulo del contorno

    cv2.imshow("edge image: "+image_filenames[index], compare)
    #cv2.imshow("license plate: ", crp_image)
    index=image_control(index)
    if(index==-1):
        break;

    #cv2.imshow(image)
image_contour.append(crp_image)

# Hay dos posibles caminos que tomar respecto a lo que ha dicho Sergio:
# El primero sería dejar como está el código (incluso quitando la ecualizacion) y hacer a posteriori otro preprocesado
# de la imagen recortada "crp_image" para intentar discernir cual es matrícula y cual no
# El segundo sería cambiar el código para primero intentar reconocer la zona donde pueda estar la matrícula y luego
# hacer un preprocesado de esa zona y buscar la matrícula ahi con el código que ya está hecho

# Yo la segunda igual la veo más liosa, como tu veas Mr. Juanjosito