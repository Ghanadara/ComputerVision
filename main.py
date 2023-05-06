from tkinter import filedialog, messagebox
import PIL
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk
import tkinter.ttk
import numpy as np
from PIL import Image
import cv2
import skimage
from skimage import img_as_ubyte

original_img = numpy.ndarray


def median_filter(img, filter_size=(3, 3), stride=1):
    img_shape = np.shape(img)
    result_shape = (int(img_shape[0] - filter_size[0] / stride + 1), int(img_shape[1] - filter_size[1] / stride + 1), 3)
    result = np.zeros(result_shape)

    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            for i in range(0, 3):
                tmp = img[h:h + filter_size[0], w:w + filter_size[1], i]
                tmp = np.sort(tmp.ravel())
                result[h, w, i] = tmp[int(filter_size[0] * filter_size[1] / 2)]
    return result


def mean_filter(img, filter_size=(3, 3), stride=1):
    img_shape = np.shape(img)

    result_shape = (int(img_shape[0] - filter_size[0] / stride + 1), int(img_shape[1] - filter_size[1] / stride + 1), 3)
    result = np.zeros(result_shape)

    for h in range(0, result_shape[0], stride):
        for w in range(0, result_shape[1], stride):
            for i in range(0, 3):
                tmp = img[h:h + filter_size[0], w:w + filter_size[1], i]
                # tmp = np.sort(tmp.ravel())
                mean = tmp.mean()
                # result[h, w, i] = tmp[int(filter_size[0] * filter_size[1] / 2)]
                result[h, w, i] = mean
    return result


def laplacian_filter(img, filter_size=3):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_img = cv2.Laplacian(grayimg, cv2.CV_8U, ksize=filter_size)
    return lap_img


def load_image():
    file = filedialog.askopenfilenames(initialdir="C:/", title="필터를 적용할 이미지를 선택하시오",
                                       filetypes=(
                                           ("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
    global original_img
    cv2_img = cv2.imread(file[0])
    original_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(original_img)
    a.axis('off')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=3)


def img_noise():
    mode = noise_combobox.get()
    if mode is not None:
        global original_img
        original_img = skimage.util.random_noise(original_img, mode=mode)
        original_img = img_as_ubyte(original_img)
        f = Figure()
        a = f.add_subplot(111)
        a.imshow(original_img)
        a.axis('off')
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=3)


def median_button():
    kernel_size = size_entry.get()
    if len(kernel_size) != 0 and int(kernel_size) >= 3:
        kernel_size = int(kernel_size)
        npimg = np.array(original_img)
        med_Img = median_filter(npimg, (kernel_size, kernel_size))
        result_Img = PIL.Image.fromarray(med_Img.astype(np.uint8))
        f = Figure()
        a = f.add_subplot(111)
        a.imshow(result_Img)
        a.axis('off')
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(column=1, row=3)
    else:
        messagebox.showerror('커널 사이즈 오류', '입력 양식 : 3이상의 정수 하나만 입력')


def mean_button():
    kernel_size = size_entry.get()
    if len(kernel_size) != 0 and int(kernel_size) >= 3:
        kernel_size = int(kernel_size)
        npimg = np.array(original_img)
        med_Img = mean_filter(npimg, (kernel_size, kernel_size))
        result_Img = PIL.Image.fromarray(med_Img.astype(np.uint8))
        f = Figure()
        a = f.add_subplot(111)
        a.imshow(result_Img)
        a.axis('off')
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(column=1, row=3)
    else:
        messagebox.showerror('커널 사이즈 오류', '입력 양식 : 3이상의 정수 하나만 입력')


def laplacian_button():
    kernel_size = size_entry.get()
    if len(kernel_size) != 0 and int(kernel_size) >= 3:
        kernel_size = int(kernel_size)
        npimg = np.array(original_img)
        laplacian_img = laplacian_filter(npimg, kernel_size)
        f = Figure()
        a = f.add_subplot(111)
        a.imshow(laplacian_img, cmap='gray')
        a.axis('off')
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(column=1, row=3)
    else:
        messagebox.showerror('커널 사이즈 오류', '입력 양식 : 3이상의 정수 하나만 입력')


root = Tk.Tk()
root_panel = Tk.Frame(root)
root_panel.grid(row=0, column=0)
btn_panel = Tk.Frame(root_panel, height=35)
btn_panel.grid(row=1, column=0)
b1 = Tk.Button(root_panel, text="그림 불러오기", command=load_image)
b2 = Tk.Button(root_panel, text="메디안 필터 실행", command=median_button)
b3 = Tk.Button(root_panel, text="평균값 필터 실행", command=mean_button)
b4 = Tk.Button(root_panel, text="라플라시안 필터 실행", command=laplacian_button)
b5 = Tk.Button(root_panel, text="노이즈 적용", command=img_noise)
b1.grid(row=0, column=0)
b2.grid(row=0, column=1)
b3.grid(row=0, column=2)
b4.grid(row=0, column=3)
b5.grid(row=0, column=4, sticky="WS")
noise = ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"]
noise_combobox = tkinter.ttk.Combobox(root_panel, values=noise)
noise_combobox.grid(row=1, column=4)
Tk.Label(root_panel, text="커널 사이즈 입력").grid(row=1, column=0, columnspan=2)
size_entry = Tk.Entry(root_panel)
size_entry.grid(row=1, column=1, columnspan=2)

root.mainloop()