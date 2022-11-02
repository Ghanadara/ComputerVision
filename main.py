from tkinter import filedialog
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk
import tkinter.ttk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import skimage


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
    lap_img = cv2.Laplacian(img, cv2.CV_8U, ksize=filter_size)
    return lap_img


def loadImage():
    file = filedialog.askopenfilenames(initialdir="C:/", title="필터를 적용할 이미지를 선택하시오",
                                       filetypes=(
                                           ("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
    global img_arr
    img_arr = PIL.Image.open(file[0])
    global cv2_img
    cv2_img = cv2.imread(file[0])
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(img_arr)
    a.axis('off')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=3)


def img_noise():
    mode = noise_combobox.get()
    if mode is not None:
        noise_img = skimage.util.random_noise(img_arr, mode=mode)
        f = Figure()
        a = f.add_subplot(111)
        a.imshow(noise_img)
        a.axis('off')
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=3)



def median_button():
    kernel_size = int(size_entry.get())
    npimg = np.array(img_arr)
    med_Img = median_filter(npimg, (kernel_size, kernel_size))
    print(type(med_Img))
    result_Img = PIL.Image.fromarray(med_Img.astype(np.uint8))
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(result_Img)
    a.axis('off')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=3)


def mean_button():
    kernel_size = int(size_entry.get())
    npimg = np.array(img_arr)
    med_Img = mean_filter(npimg, (kernel_size, kernel_size))
    print(type(med_Img))
    result_Img = PIL.Image.fromarray(med_Img.astype(np.uint8))
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(result_Img)
    a.axis('off')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=3)


def Laplacian_button():
    kernel_size = int(size_entry.get())
    cv2_img_copy = cv2_img
    result_Img = laplacian_filter(cv2_img_copy, kernel_size)
    f = Figure()
    a = f.add_subplot(111)
    a.imshow(result_Img)
    a.axis('off')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=3)


root = Tk.Tk()
root_panel = Tk.Frame(root)
# root_panel.pack(side="bottom", fill="both", expand="yes")
root_panel.grid(row=0, column=0)

btn_panel = Tk.Frame(root_panel, height=35)
# btn_panel.pack(side='top', fill="both", expand="yes")
btn_panel.grid(row=1, column=0)
# img_arr = mpimg.imread('165958619369.jpg')
# imgplot = plt.imshow(img_arr)
b1 = Tk.Button(root_panel, text="그림 불러오기", command=loadImage)
b2 = Tk.Button(root_panel, text="메디안 필터 실행", command=median_button)
b3 = Tk.Button(root_panel, text="평균값 필터 실행", command=mean_button)
b4 = Tk.Button(root_panel, text="라플라시안 필터 실행", command=laplacian_filter)
b5 = Tk.Button(root_panel,text="노이즈 적용",command=img_noise)
b1.grid(row=0, column=0)
b2.grid(row=0, column=1)
b3.grid(row=0, column=2)
b4.grid(row=0, column=3)
b5.grid(row=0, column=4, sticky="WS")
noise = ["gaussian", "localvar", "poisson", "salt", "pepper", "salt&pepper", "speckle"]
noise_combobox = tkinter.ttk.Combobox(root_panel, values=noise)
noise_combobox.grid(row=1, column=4)
Tk.Label(root_panel, text="커널 사이즈 입력").grid(row=1, column=0,columnspan=2)
size_entry = Tk.Entry(root_panel)
size_entry.grid(row=1, column=1,columnspan=2)

#
# # image 1

# npimg_1 = np.array(img_1)
# plt.imshow(npimg_1)
# #med_img_1 = median_filter(npimg_1)
# # show(med_img_1, max_ylim=12500)
#
# #med_img_1 = median_filter(npimg_1, (5, 5))
# # show(med_img_1, max_ylim=12500)
#
# # image 2
# # img_2 = Image.open('165958619369.jpg').convert('L')
# # npimg_2 = np.array(img_2)
# # show(npimg_2, max_ylim=4500)
# #
# # med_img_2 = median_filter(npimg_2)
# # show(med_img_2, max_ylim=4500)
# #
# # med_img_2 = median_filter(npimg_2, (5, 5))
# # show(med_img_2, max_ylim=4500)
# # fig = plt.show(npimg_1, max_ylim=12500)
# # root = Tk.Tk()
# # label = Tk.Label(root, text="라벨").grid(column=0, row=0)
# # canvas = FigureCanvasTkAgg(fig,master=root)
# # canvas.get_tk_widget().grid(column=0, row=1)
# #
# # Tk.mainloop()

# canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
# canvas._tkcanvas.pack(side="top", fill="both", expand=1)
root.mainloop()
