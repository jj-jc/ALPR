from tkinter import *
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import filedialog
import webbrowser
import cv2
import glob
import numpy as np
import pytesseract as tess

tess.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

global dir_entry
global im_frame
global language
global option
global menubar
global eng_menubar
global dir_label
global prev_label
global mat_label
global enc_label
global preg_label
global mat2_label
global autor_label
global ver_label
global mat_btn
global dir_btn
global enc_btn
global button_forward
global button_back
global index
global status
global archivo

index=0

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

def ayuda_menu():
    webbrowser.open_new(r'help.pdf')
    return

def forward(im_number):
    global index
    index += 1
    button_forward.config(command=lambda: forward(im_number+1))
    button_back.config(command=lambda: back(im_number-1))
    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(results[0][im_number-1], cv2.COLOR_BGR2RGB)))
    img_label.config(image=img)
    img_label.image = img
    mat2_label.config(text=results[1][im_number-1])
    button_back.config(state=NORMAL)
    if im_number==len(results[0]):
        button_forward.config(state=DISABLED)
    status.config(text=str(im_number) + "/" + str(len(results[0])))

    return

def back(im_number):
    global index
    index -= 1
    button_forward.config(command=lambda: forward(im_number+1))
    button_back.config(command=lambda: back(im_number-1))
    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(results[0][im_number-1], cv2.COLOR_BGR2RGB)))
    img_label.config(image=img)
    img_label.image = img
    mat2_label.config(text=results[1][im_number-1])
    button_back.config(state=NORMAL)
    if im_number==1:
        button_back.config(state=DISABLED)
    status.config(text=str(im_number) + "/" + str(len(results[0])))
    return

def click_imagen():
    global img_label
    mat_btn.config(state=NORMAL)
    for widget in im_frame.winfo_children():
        widget.destroy()

    if option.get()==0:
        root.filename = filedialog.askopenfilename(initialdir="C:/", title="Elegir imágen",
                                                   filetypes=(("jpg", "*.jpg"), ("png", "*.png"), ("todos", "*.*"),
                                                              ("gif", "*.gif")))
        dir_entry.delete(0, END)
        dir_entry.insert(0, root.filename)
        img = ImageTk.PhotoImage(Image.open(root.filename))
        img_label = Label(im_frame, image=img)
    else:
        root.filename = filedialog.askdirectory(initialdir="C:/", title="Elegir carpeta")
        dir_entry.delete(0, END)
        dir_entry.insert(0, root.filename + "/")
        image_filenames = glob.glob(dir_entry.get() + "*.*")
        aux_im=cv2.imread(image_filenames[0])
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(aux_im, cv2.COLOR_BGR2RGB)))
        img_label = Label(im_frame, image=img)
        status.config(text=str(1) + "/" + str(len(image_filenames)))

    width = img.width()
    height = img.height()
    root.minsize(width + 120, height + 120)
    img_label.image = img
    img_label.pack()

    return

def algoritmo():
    global results
    global dilated_matrix
    global edges_matrix
    global img_box_matrix
    global image_contour_matrix

    img_matrix=[]
    results=[[],[]]
    dilated_matrix=[]
    edges_matrix=[]
    img_box_matrix=[]
    image_contour_matrix=[]

    if option.get()==0:
        img_matrix.append(cv2.imread(dir_entry.get()))
    else:
        image_filenames = glob.glob(dir_entry.get() + "/*.*")
        for image in image_filenames:
            img_matrix.append(cv2.imread(image))

    for img_mat in img_matrix:
        gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(median)
        binary = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        edges = cv2.Laplacian(dilated, cv2.CV_8UC1)

        cont, hier = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_contours = img_mat.copy()
        cv2.drawContours(all_contours, cont, -1, (255, 0, 0), 3)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[:8]
        largest_extent = 0
        edges_aux = edges.copy()

        for i in cont:
            x, y, w, h = cv2.boundingRect(i)
            aspect_ratio = w / h
            if 7 > aspect_ratio > 1.3:
                extent = cv2.contourArea(i) / (w * h)
                if extent > largest_extent:
                    largest_extent = extent
                    crp_image = img_mat[y:y + h, x:x + w]

                    img_rect = cv2.rectangle(img_mat, (x, y), (x + w, y + h), (0, 0, 255), 2)
            img_box = cv2.rectangle(edges_aux, (x, y), (x + w, y + h), (255, 0, 0), 2)

        gray_crop = cv2.cvtColor(crp_image, cv2.COLOR_BGR2GRAY)  # convert it to grayscale

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_crop = clahe.apply(gray_crop)  # equalize crop

        binarized_crop = (cv2.threshold(equalized_crop, 50, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])  # Binarization using OTSU method
        filtered_crop = cv2.medianBlur(binarized_crop, 5)  # Median filter the crop
        kernel = np.ones((3, 3), np.uint8)  # Kernel for closing the crop
        closed_crop = cv2.morphologyEx(filtered_crop, cv2.MORPH_CLOSE, kernel)  # Closing the crop
        kernel = np.ones((3, 3), np.uint8)  # Kernel for dilation the crop
        dilated_crop = cv2.dilate(closed_crop, kernel, iterations=1)  # Dilating the crop
        kernel = np.ones((3, 3), np.uint8)  # Kernel for erode the crop
        eroded_crop = cv2.erode(dilated_crop, kernel, iterations=1)  # Erode the crop
        mask = createBorderMask(eroded_crop, 5)  # Mask for reduce external border of the crop
        masked_crop = cv2.add(eroded_crop, mask)  # Apply mask
        kernel = np.ones((3, 3), np.uint8)  # Kernel for closing
        closed_masked_crop = cv2.morphologyEx(masked_crop, cv2.MORPH_CLOSE,
                                              kernel)  # Close masked crop
        plate_number = tess.image_to_string(closed_masked_crop,
                                            config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 8 --oem 3')

        dilated_matrix.append(dilated)
        edges_matrix.append(edges)
        img_box_matrix.append(img_box)
        image_contour_matrix.append(closed_masked_crop)
        results[0].append(img_rect)
        results[1].append(plate_number[:-2])

    mat2_label.config(text=results[1][0])
    img_rect2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(results[0][0], cv2.COLOR_BGR2RGB)))
    width = img_rect2.width()
    height = img_rect2.height()
    root.minsize(width + 120, height + 120)
    img_label.config(image=img_rect2)
    img_label.image = img_rect2
    return

def pasos():
    pasos_win = Toplevel()
    pasos_win.resizable(False, False)
    pasos_win.iconbitmap('computer-vision.ico')

    if language.get() == 0:
        pasos_win.title("Paso a Paso")
    else:
        pasos_win.title("Step by Step")

    aux1 = Image.fromarray(cv2.cvtColor(dilated_matrix[index], cv2.COLOR_GRAY2RGB))
    aux2 = Image.fromarray(cv2.cvtColor(edges_matrix[index], cv2.COLOR_GRAY2RGB))
    aux3 = Image.fromarray(cv2.cvtColor(img_box_matrix[index], cv2.COLOR_GRAY2RGB))
    aux1 = aux1.resize((int(aux1.width/2), int(aux1.height/2)))
    aux2 = aux2.resize((int(aux2.width/2), int(aux2.height/2)))
    aux3 = aux3.resize((int(aux3.width/2), int(aux3.height/2)))
    img1 = ImageTk.PhotoImage(aux1)
    img2 = ImageTk.PhotoImage(aux2)
    img3 = ImageTk.PhotoImage(aux3)
    img4 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_contour_matrix[index], cv2.COLOR_BGR2RGB)))

    paso1 = Label(pasos_win, image=img1)
    paso2 = Label(pasos_win, image=img2)
    paso3 = Label(pasos_win, image=img3)
    paso4 = Label(pasos_win, image=img4)
    paso1.image = img1
    paso1.grid(row=0, column=0)
    paso2.image = img2
    paso2.grid(row=0, column=1)
    paso3.image = img3
    paso3.grid(row=1, column=0)
    paso4.image = img4
    paso4.grid(row=1, column=1)

def config(window):
    emptymenu = Menu(root)
    root.config(menu=emptymenu)

    if language.get() == 0:
        root.title('Escáner de Matrículas')
        root.config(menu=menubar)
        dir_label.config(text="Ruta de la Imágen")
        prev_label.config(text="Previsualización")
        mat_label.config(text="Lectura de Matrícula")
        enc_label.config(text="")
        preg_label.config(text="")
        mat_btn.config(text="Iniciar")
        dir_btn.config(text="Importar")
        enc_btn.config(text="Enviar")

    else:
        root.title('Automatic Licence Plate Recognition')
        root.config(menu=eng_menubar)
        dir_label.config(text="Image Path")
        prev_label.config(text="Preview")
        mat_label.config(text="Licence Plate Reading")
        mat_btn.config(text="Begin")
        dir_btn.config(text="Import")
        enc_btn.config(text="Send")

    if option.get()==0:
        status.grid_forget()
        autor_label.grid_forget()
        ver_label.grid_forget()
        button_forward.grid_forget()
        button_back.grid_forget()
        autor_label.grid(row=5, column=1, sticky=E, pady=(4, 0))
        ver_label.grid(row=5, column=0, sticky=W, pady=(4, 0))
    else:
        autor_label.grid_forget()
        ver_label.grid_forget()
        status.grid(row=5, column=0, pady=(4,0))
        button_forward.grid(row=5, column=0, sticky=E, padx=(0, 6), pady=(4,0))
        button_back.grid(row=5, column=0, sticky=W, pady=(4,0))
        autor_label.grid(row=6, column=1, sticky=E, pady=(4, 0))
        ver_label.grid(row=6, column=0, sticky=W, pady=(4, 0))

    window.destroy()
    return


def callback(url):
    webbrowser.open_new(url)
    return


def acerca_de():
    info = Toplevel()
    info.resizable(False, False)
    info.iconbitmap('computer-vision.ico')

    if language.get() == 0:
        info.title("Acerca de...")
        img2 = ImageTk.PhotoImage(Image.open('acercade.jpg'))
        img_label2 = Label(info, image=img2)
    else:
        info.title("About us...")
        img2 = ImageTk.PhotoImage(Image.open('about-us.jpg'))
        img_label2 = Label(info, image=img2)

    img_label2.image = img2
    img_label2.bind("<Button-1>", lambda e: callback("https://alpr-system8.webnode.es/"))

    img_label2.pack()
    return


def prefs():
    if language.get() == 0:
        pref_window = Toplevel()
        pref_window.geometry("200x200")
        pref_window.resizable(False, False)
        pref_window.title("Preferencias")
        pref_window.iconbitmap('computer-vision.ico')

        pref_window.columnconfigure(0, weight=1)
        pref_window.rowconfigure(0, weight=1)
        pref_window.rowconfigure(1, weight=1)

        language_frame = ttk.LabelFrame(pref_window, text="Idioma", borderwidth=2, relief="ridge")
        option_frame = ttk.LabelFrame(pref_window, text="Tipo de importación", borderwidth=2, relief="ridge")

        lang1 = ttk.Radiobutton(language_frame, text="Español", variable=language, value=0)
        lang2 = ttk.Radiobutton(language_frame, text="Inglés", variable=language, value=1)
        option1 = ttk.Radiobutton(option_frame, text="Simple", variable=option, value=0)
        option2 = ttk.Radiobutton(option_frame, text="Múltiple", variable=option, value=1)

        ok_btn = ttk.Button(pref_window, text="Aceptar", command=lambda: config(pref_window))
        cancel_btn = ttk.Button(pref_window, text="Cancelar", command=pref_window.destroy)

        language_frame.grid(row=0, column=0, columnspan=2, sticky=(W, E), pady=5, padx=5)
        option_frame.grid(row=1, column=0, columnspan=2, sticky=(W, E), pady=(0, 5), padx=5)
        ok_btn.grid(row=2, column=0, pady=(0, 5), padx=5, sticky=W)
        cancel_btn.grid(row=2, column=1, pady=(0, 5), padx=(0, 5))

        lang1.grid(row=0, column=0, sticky=(W, E))
        lang2.grid(row=1, column=0, sticky=(W, E))
        option1.grid(row=0, column=0, sticky=(W, E))
        option2.grid(row=1, column=0, sticky=(W, E))
    else:
        pref_window = Toplevel()
        pref_window.geometry("200x200")
        pref_window.resizable(False, False)
        pref_window.title("Preferences")
        pref_window.iconbitmap('computer-vision.ico')

        pref_window.columnconfigure(0, weight=1)
        pref_window.rowconfigure(0, weight=1)
        pref_window.rowconfigure(1, weight=1)

        language_frame = ttk.LabelFrame(pref_window, text="Language", borderwidth=2, relief="ridge")
        option_frame = ttk.LabelFrame(pref_window, text="Import mode", borderwidth=2, relief="ridge")

        lang1 = ttk.Radiobutton(language_frame, text="Spanish", variable=language, value=0)
        lang2 = ttk.Radiobutton(language_frame, text="English", variable=language, value=1)
        option1 = ttk.Radiobutton(option_frame, text="Single", variable=option, value=0)
        option2 = ttk.Radiobutton(option_frame, text="Multiple", variable=option, value=1)

        ok_btn = ttk.Button(pref_window, text="Ok", command=lambda: config(pref_window))
        cancel_btn = ttk.Button(pref_window, text="Cancel", command=pref_window.destroy)

        language_frame.grid(row=0, column=0, columnspan=2, sticky=(W, E), pady=5, padx=5)
        option_frame.grid(row=1, column=0, columnspan=2, sticky=(W, E), pady=(0, 5), padx=5)
        ok_btn.grid(row=2, column=0, pady=(0, 5), padx=5, sticky=W)
        cancel_btn.grid(row=2, column=1, pady=(0, 5), padx=(0, 5))

        lang1.grid(row=0, column=0, sticky=(W, E))
        lang2.grid(row=1, column=0, sticky=(W, E))
        option1.grid(row=0, column=0, sticky=(W, E))
        option2.grid(row=1, column=0, sticky=(W, E))
    return


root = Tk()
root.minsize(600, 400)

root.title('Escáner de Matrículas')
root.iconbitmap('computer-vision.ico')

respuesta = IntVar()
language = IntVar()
option = IntVar()
language.set(0)
option.set(0)

menubar = Menu(root)

root.config(menu=menubar)
archivo = Menu(menubar, tearoff=0)
archivo.add_command(label="Ver Pasos", command=pasos)
archivo.add_separator()
archivo.add_command(label="Salir", command=root.quit)
editar = Menu(menubar, tearoff=0)
editar.add_command(label="Preferencias", command=prefs)
ayuda = Menu(menubar, tearoff=0)
ayuda.add_command(label="Ayuda", command=ayuda_menu)
ayuda.add_separator()
ayuda.add_command(label="Acerca de...", command=acerca_de)
menubar.add_cascade(label="Archivo", menu=archivo)
menubar.add_cascade(label="Editar", menu=editar)
menubar.add_cascade(label="Ayuda", menu=ayuda)

eng_menubar = Menu(root)

archivo = Menu(eng_menubar, tearoff=0)
archivo.add_command(label="Step by Step", command=pasos)
archivo.add_separator()
archivo.add_command(label="Close", command=root.quit)
editar = Menu(eng_menubar, tearoff=0)
editar.add_command(label="Preferences", command=prefs)
ayuda = Menu(eng_menubar, tearoff=0)
ayuda.add_command(label="Help", command=ayuda_menu)
ayuda.add_separator()
ayuda.add_command(label="About us...", command=acerca_de)
eng_menubar.add_cascade(label="File", menu=archivo)
eng_menubar.add_cascade(label="Edit", menu=editar)
eng_menubar.add_cascade(label="Help", menu=ayuda)

content = ttk.Frame(root, padding=(6, 6, 6, 1))
im_frame = ttk.Frame(content, borderwidth=5, relief="ridge", height=400, width=400)
enc_frame = ttk.Frame(content, borderwidth=5, relief="ridge")
lect_frame = ttk.Frame(content)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
content.columnconfigure(0, weight=1)
content.columnconfigure(1, weight=0)
content.rowconfigure(3, weight=1)

dir_label = ttk.Label(content, text="Ruta de la Imágen", font='Tahoma 10')
prev_label = ttk.Label(content, text="Previsualización", font='Tahoma 10')
mat_label = ttk.Label(lect_frame, text="Lectura de Matrícula", font='Tahoma 10')
enc_label = ttk.Label(enc_frame, text="Encuesta")
preg_label = ttk.Label(enc_frame, text="¿Ha sido correcta la lectura?")
mat2_label = ttk.Label(lect_frame, text="0000AAA", font='Tahoma 10 bold', borderwidth=5, relief="ridge",
                       anchor="center")
autor_label = ttk.Label(content, text="powered by ALPR System", font='Tahoma 8 italic', anchor="e")
ver_label = ttk.Label(content, text="ver. 1.0", font='Tahoma 8 italic', anchor="w")
status = Label(content, text=str(0) + "/" + str(0), anchor="c")

dir_entry = ttk.Entry(content)

mat_btn = ttk.Button(lect_frame, text="Iniciar", command=algoritmo, state=DISABLED)
dir_btn = ttk.Button(content, text="Importar", command=click_imagen)
enc_btn = ttk.Button(enc_frame, text="Enviar")
button_forward = Button(content, text=">>", command=lambda: forward(2))
button_back = Button(content, text="<<", command=back, state=DISABLED)

op1 = ttk.Radiobutton(enc_frame, text="Sí", variable=respuesta, value=1)
op2 = ttk.Radiobutton(enc_frame, text="No", variable=respuesta, value=2)
op3 = ttk.Radiobutton(enc_frame, text="NS/NC", variable=respuesta, value=3)

content.grid(row=0, column=0, sticky=(N, S, E, W))
dir_label.grid(row=0, column=0, columnspan=2, sticky=W, pady=(0, 5))
dir_btn.grid(row=1, column=1, sticky=(W, E))
dir_entry.grid(row=1, column=0, sticky=(W, E), padx=(0, 6))
prev_label.grid(row=2, column=0, columnspan=2, sticky=W, pady=(10, 5))
im_frame.grid(row=3, column=0, rowspan=2, sticky=(N, S, W, E), padx=(0, 6))
autor_label.grid(row=5, column=1, sticky=E, pady=(4, 0))
ver_label.grid(row=5, column=0, sticky=W, pady=(4, 0))

lect_frame.grid(row=3, column=1, sticky=(N, W, E))
# enc_frame.grid(row=4, column=1, sticky=(S,W,E))

mat_label.pack(fill=X, pady=(0, 10))
mat_btn.pack(fill=X)
mat2_label.pack(pady=(10, 0), ipady=4, ipadx=4)

enc_label.grid(row=0, column=0, sticky=W)
preg_label.grid(row=1, column=0, sticky=W)
op1.grid(row=2, column=0, sticky=W)
op2.grid(row=3, column=0, sticky=W)
op3.grid(row=4, column=0, sticky=W)
enc_btn.grid(row=5, column=0, sticky=(N, S, W, E))

root.mainloop()
