#!/usr/bin/env python3

# input
curve = "A"
interactive_plot = "Contour"
image_path = "Prescale_MS_Loz.jpg"
Curve_path = "LLW_Prescale_A_B_C_D_2min.csv"
f_scale = 5
f_smooth = 10

import cv2
import numpy
import plotly.graph_objects as go

import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

import csv

Curve_density = []
Curve_pressure = []

if curve == "A":
    IDX = 0
elif curve == "B":
    IDX = 2
elif curve == "C":
    IDX = 4
elif curve == "D":
    IDX = 6
else:
    print("Curve letter not recongnised.. Choosing A instead")

with open(Curve_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            Curve_pressure.append(float(row[IDX]))
            Curve_density.append(float(row[IDX+1]))
        except:
            None

# ABCD curve data into callable
f_chart = interpolate.interp1d(Curve_density, Curve_pressure, fill_value="extrapolate")

# read image
RGB_img = cv2.imread(image_path)
h, w = RGB_img.shape[0], RGB_img.shape[1]

# convertion to Hue, Saturation, Value color space
HSV_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
kernel = numpy.ones((f_smooth,f_smooth),numpy.float32)/(f_smooth**2)
HSV_img = cv2.filter2D(HSV_img,-1,kernel)

# color detection
lower_bound = numpy.array([125,10,150])
upper_bound = numpy.array([135,255,255])
reddish_mask = cv2.inRange(HSV_img, lower_bound, upper_bound)
reddishHSV_img = cv2.bitwise_and(HSV_img, HSV_img, mask=reddish_mask)
reddishHSV_img = cv2.resize(reddishHSV_img, (int(h/f_scale),int(w/f_scale)))

# reading out the saturatiion of selected color range
redish_saturation = reddishHSV_img[:,:,1]
max_saturation = numpy.max(redish_saturation)
print(f"maximum saturation: {max_saturation}")
Density = redish_saturation*(1.1/229)
Pressure = f_chart(Density)
print(Pressure.shape)


plt.contourf(numpy.flip(Pressure, axis=0))
plt.colorbar()
plt.savefig("output_" + image_path + ".jpg", figsize=(12, 12), dpi=600)

if interactive_plot == "Contour":
    fig = go.Figure(data=[go.Contour(z=Pressure)])
    fig.show()
elif interactive_plot == "Surface":
    fig = go.Figure(data=[go.Surface(z=Pressure)])
    fig.show()
