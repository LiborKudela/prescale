#!/usr/bin/env python3
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import split
import cv2
import numpy
import scipy.interpolate as interpolate
import csv

description = """
Estimate contact pressure from imgage of a prescale foil

Examples of use:
python3 readfoil.py test_files/circle.jpg
python3 readfoil.py test_files/circle.jpg -viz=i2D, -rescale=5 -curve=B
"""

parser = ArgumentParser(description=description,
                        formatter_class=RawTextHelpFormatter)

parser.add_argument("image_path", help="mandatory")
parser.add_argument("-sheet",
                    dest="sheet",
                    default="LLW_2min",
                    help="selection of sheet table (default=LLW_2min)")
parser.add_argument("-curve",
                    dest="curve",
                    choices=["A","B","C","D"],
                    default="A",
                    help="selection of curve from the sheet table (default=A)")
parser.add_argument("-viz",
                    dest="viz",
                    choices=["none","2D","i2D","i3D"],
                    default="2D",
                    help="i corresponds to interactive versions (default=2D)")
parser.add_argument("-smooth",
                    dest="f_smooth",
                    type=int,
                    default=1,
                    help="kernel size for smoothing input data (default=1)")
parser.add_argument("-rescale",
                    dest="f_scale",
                    type=int,
                    default=1,
                    help="lowers resolution of output by factor, speeds up interactive plots (default=1)")
args = parser.parse_args()

head, tail = split(args.image_path)
print("Evaluating: " + tail)
print(f"Applling sheet tables:     {args.sheet}")

curve_density = []
curve_pressure = []
IDX = {"A": 0, "B": 2, "C": 4, "D": 6}
with open("sheet_tables/" + args.sheet + ".csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for row in reader:
        try:
            curve_pressure.append(float(row[IDX[args.curve]]))
            curve_density.append(float(row[IDX[args.curve]+1]))
        except:
            None

# ABCD curve data into callable
f_chart = interpolate.interp1d(curve_density, curve_pressure, fill_value="extrapolate")

# read image
RGB_img = cv2.imread(args.image_path)
h, w = RGB_img.shape[0], RGB_img.shape[1]

# convertion to Hue, Saturation, Value color space
HSV_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HSV)
kernel = numpy.ones((args.f_smooth,args.f_smooth),numpy.float32)/(args.f_smooth**2)
HSV_img = cv2.filter2D(HSV_img,-1,kernel)

# color detection
lower_bound = numpy.array([125,10,150])
upper_bound = numpy.array([135,255,255])
reddish_mask = cv2.inRange(HSV_img, lower_bound, upper_bound)
reddishHSV_img = cv2.bitwise_and(HSV_img, HSV_img, mask=reddish_mask)
reddishHSV_img = cv2.resize(reddishHSV_img, (int(h/args.f_scale),int(w/args.f_scale)))

# reading out the saturatiion of selected color range
redish_saturation = reddishHSV_img[:,:,1]
density = redish_saturation*(1.1/229)
print("Max density estimation:    %.3f" % numpy.max(density))
pressure = numpy.flip(f_chart(density), axis=0)

max_pressure_pixel = numpy.unravel_index(numpy.argmax(pressure), pressure.shape)
max_pressure = pressure[max_pressure_pixel]
print("Max pressure estimation:   %.3f MPa" % max_pressure)
print(f"Max pressure position:     {max_pressure_pixel}")
print(f"Results resolution:        {pressure.shape}")

# visualisation
if args.viz != "none":
    plot_file = f"{args.viz}_plot_" + tail[:-4]
    if args.viz == "2D":
        import matplotlib.pyplot as plt
        plt.contourf(pressure)
        plt.colorbar()
        filename = plot_file + ".jpg"
        plt.savefig(filename, figsize=(12, 12), dpi=600)
        print(f"Static visual output: {filename}")
    elif args.viz == "i2D":
        import plotly.graph_objects as go
        from plotly.offline import plot
        fig = go.Figure(data=[go.Contour(z=pressure)])
        filename = plot_file + ".html"
        plot(fig, filename=filename, auto_open=False)
        print(f"Interactive visual output: {filename} (open in a web browser)")
    elif args.viz == "i3D":
        import plotly.graph_objects as go
        from plotly.offline import plot
        fig = go.Figure(data=[go.Surface(z=pressure)])
        filename = plot_file + ".html"
        plot(fig,filename=filename, auto_open=False)
        print(f"Interactive visual output: {filename} (open in a web browser)")
