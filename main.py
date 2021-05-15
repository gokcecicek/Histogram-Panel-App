'''
@author: Beril Gökçe Çiçek
'''#

import tkinter as tk
from tkinter import *
import cv2
import numpy as np
from PIL import ImageTk, Image
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter import Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HistogramPanel:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Main Panel")
        self.window.geometry("900x600")

        self.img_arr = []
        self.imageCanvas = Canvas(self.window, height=280, width=280, bg="#f0f0f0")
        self.imageCanvas.place(x=200, y=20)

        self.desiredCanvas = Canvas(self.window, height=280, width=280, bg="#f0f0f0")
        self.desiredCanvas.place(x=600, y=20)

        c = Canvas(self.window, width=150, height=700, bg="#d1ccc0")  # menuCanvas
        c.place(x=0, y=0)

        self.selectImgButton = Button(self.window, text="Select image", command=self.selectImageEvent,
                                      activebackground="#aaa69d", width=12)
        self.selectImgButton.pack(side=LEFT, padx="10", pady="10")
        self.selectImgButton.place(x=30, y=50)

        self.desiredImgButton = Button(self.window, text="Desired image", command=self.desiredImageEvent,
                                       activebackground="#aaa69d", width=12)
        self.desiredImgButton.pack(side=LEFT, padx="10", pady="10")
        self.desiredImgButton.place(x=30, y=100)

        self.histogramButton = Button(self.window, text="Histogram", command=self.histogramEvent,
                                      activebackground="#aaa69d", width=12)
        self.histogramButton.pack(side=LEFT, padx="10", pady="10")
        self.histogramButton.place(x=30, y=150)

        self.equalButton = Button(self.window, text="Equalization", command=self.equalizationEvent,
                                  activebackground="#aaa69d", width=12)
        self.equalButton.pack(side=LEFT, padx="10", pady="10")
        self.equalButton.place(x=30, y=200)

        self.clearButton = Button(self.window, text="Clear Images", command=self.clearAllEvent,
                                  activebackground="#aaa69d", width=12)
        self.clearButton.pack(side=LEFT, padx="10", pady="10")
        self.clearButton.place(x=30, y=250)

        self.exitButton = Button(self.window, text="Exit", command=self.exitEvent,
                                 activebackground="#aaa69d", width=12)
        self.exitButton.pack(side=LEFT, padx="10", pady="10")
        self.exitButton.place(x=30, y=300)

        self.window.mainloop()

    def selectImageEvent(self):
        self.main_path = filedialog.askopenfilename()
        if len(self.main_path) > 0:
            self.selected_image = Image.open(self.main_path).convert('YCbCr')
            self.img_arr.append(self.selected_image)
            self.selected_image = ImageTk.PhotoImage(self.selected_image)
            self.selected_image = np.array(self.selected_image)
            self.imageCanvas.image = self.selected_image
            self.imageCanvas.create_image(0, 0, anchor='nw', image=self.selected_image)

    def desiredImageEvent(self):
        self.desired_path = filedialog.askopenfilename()
        if len(self.desired_path) > 0:
            image = cv2.imread(self.desired_path)
            self.img_arr.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.desiredCanvas.image = image
            self.desiredCanvas.create_image(0, 0, anchor='nw', image=image)

    def histogramEvent(self):
        self.count=0
        hist_window = tk.Tk()
        hist_window.title("Histogram Panel")
        hist_window.geometry("900x600")
        histogramCanvas1 = Canvas(hist_window, height=600, width=600, bg="#f0f0f0")
        histogramCanvas1.place(x=50, y=50)
        histogramCanvas2 = Canvas(hist_window, height=600, width=600, bg="#f0f0f0")
        histogramCanvas2.place(x=700, y=50)

        main_item = self.img_arr[0]
        main_item = np.array(main_item)
        main_item = main_item[:, :, 0:1].flatten()  # Convert 1-D array
        desired_item = self.img_arr[1]

        fig1 = self.createHistogram(main_item)
        fig2 = self.createHistogram(desired_item)
        chart1 = FigureCanvasTkAgg(fig1, histogramCanvas1)
        chart2 = FigureCanvasTkAgg(fig2, histogramCanvas2)
        chart1.get_tk_widget().pack()
        chart2.get_tk_widget().pack()
        # fig.show()

    def createHistogram(self, item):
        hist = np.zeros([256])  # 256 gray value
        for i in range(item.shape[0]):
            value = item[i]
            hist[value] += 1
        fig = plt.figure(figsize=(6, 6), dpi=100)
        f = fig.add_subplot(111)
        f.bar(np.arange(len(hist)), hist)
        f.set_xlabel('Gray level')
        f.set_ylabel('Number of pixel(Counting)')
        if self.count == 0:
            f.set_title('Histogram of selected image')
            self.selected_hist = hist
        if self.count == 1:
            f.set_title('Histogram of desired image')
            self.desired_hist = hist
        self.count += 1
        return fig

    def equalizationEvent(self):
        equal_window = Toplevel()
        equal_window.title("Equalization Panel")
        equal_window.geometry("900x600")
        equalCanvas = Canvas(equal_window, height=280, width=280, bg="#f0f0f0")
        equalCanvas.place(x=50, y=20)
        equalHistogramCanvas = Canvas(equal_window, height=280, width=280, bg="#f0f0f0")
        equalHistogramCanvas.place(x=500, y=20)

        colorful_img = self.img_arr[0]
        width, height = colorful_img.size
        img = np.array(colorful_img)

        cumulative = self.calculateCumulative(self.selected_hist) # Calculate CDF
        # Calculate (L-1)*CDF and round operation part
        result = np.zeros(256, dtype=int)
        Gray_Level = 256
        for i in range(self.selected_hist.size):
            result[i] = round(((Gray_Level-1)* cumulative[i]) / (width * height))

        new_image = img.copy()
        new_image[:, :, 0] = list(map(lambda a: result[a], img[:, :, 0]))

        fig = plt.figure(figsize=(6, 6), dpi=100)
        f = fig.add_subplot(111)
        f.bar(np.arange(len(result)), result)
        f.set_xlabel('Gray level')
        f.set_ylabel('Number of pixel(counting)')
        f.set_title('Histogram Equalization')
        chart = FigureCanvasTkAgg(fig, equalHistogramCanvas) # Add equalization histogram to canvas
        chart.get_tk_widget().pack()

        output_image = Image.fromarray(np.uint8(new_image), "YCbCr") # Equal image
        output_image = ImageTk.PhotoImage(output_image)
        equalCanvas.image = output_image # Add output image to canvas
        equalCanvas.create_image(0, 0, anchor='nw', image=output_image)

    def calculateCumulative(self, hist): # Calculate CDF
        cum_arr = np.zeros(256, dtype=int)
        cum_arr[0] = hist[0]
        for i in range(1, hist.size):
            cum_arr[i] = cum_arr[i - 1] + hist[i]
        return cum_arr


    def clearAllEvent(self):
        self.imageCanvas.delete("all")
        self.desiredCanvas.delete("all")

    def exitEvent(self):
        self.window.destroy()

gui = HistogramPanel()
