# -*- coding: utf-8 -*-



if 'google.colab' in str(get_ipython()):
  !pip install git+https://github.com/FullControlXYZ/fullcontrol --quiet
import fullcontrol as fc

import numpy as np
import cv2
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from shapely.geometry import Point

"""# Image-to-spiral utils"""

def get_failure(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_faulure_contours(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.bitwise_not(thresh, thresh)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))

    # find the convex hull object for each contour
    hull_list = []
    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        hull_list.append(hull)

    for i in range(len(cnts)):
        cv2.drawContours(img, [cnts[i]], -1, (0,255,0), 4)
        cv2.drawContours(img, hull_list, i, (0,0,255), 2)

    all_in_one_contours = np.vstack(cnts)
    all_in_one_hull_points = cv2.convexHull(all_in_one_contours)
    print("number of points in the big hull = ", len(all_in_one_hull_points))

    cv2.drawContours(img, [all_in_one_hull_points], 0, (255,0,0), 2)

    # find the middle point
    c = max([all_in_one_hull_points], key=cv2.contourArea)
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    distance = np.sqrt( (right[0] - left[0])**2 + (right[1] - left[1])**2 )
    x,y,w,h = cv2.boundingRect(c)
    centx = np.sqrt( ((right[0] + left[0])**2)/4)
    centy = np.sqrt( ((right[1] + left[1])**2)/4 )
    print('Center coords = ',centx, centy)
    #origin = [centx,centy]

    #cv2.line(img, left, right, (255,0,0), 2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img, all_in_one_hull_points, centx, centy


def generate_continuous_spiral(number_of_segments, sparce_koefficient, hull_points, centx, centy):
    ray = 0
    points_buffer = []
    spiral_points = []

    for hp in range(len(hull_points)):
        if(hp%sparce_koefficient == 0):
            ray += 1
            a = hull_points.T[0][0][hp] # running x
            b = hull_points.T[1][0][hp] # running y
            line_len = np.sqrt((centx-a)**2 + (centy-b)**2)
            seg_len = np.round(line_len/number_of_segments,2)-0.01 # epsilon=0.1

            # generate concentric circles around the given point
            p = Point(a,b)
            idx = 0
            for seg in range(1,number_of_segments):
                idx += 1
                c = p.buffer(seg_len*seg).boundary
                l = LineString([(a,b),(centx,centy)])
                seg = c.intersection(l)
                points_buffer.append((seg.coords[0][0],seg.coords[0][1]))

    # rearrnge the points to create a continuous spiral
    for spiral in range(number_of_segments-1):
        for point in points_buffer[spiral::number_of_segments-1]:
            spiral_points.append(point)

    return spiral_points

"""# FullControl utils"""

def generate_gcode_steps(spiral_points, z_height, layer_height, img_to_real_scale, img_to_real_origin):
    steps = []
    for i in range(len(spiral_points)):
        steps.append(fc.Point(x=spiral_points[i][0]/img_to_real_scale+img_to_real_origin[0], y=spiral_points[i][1]/img_to_real_scale+img_to_real_origin[1], z=z_height+layer_height))
    return steps





"""# Main"""

filename = '/.../failure_mask1.png'

number_of_segments = 10
sparce_koefficient = 1

img, hull_points, centx, centy = get_faulure_contours(filename)
spiral_points = generate_continuous_spiral(number_of_segments, sparce_koefficient, hull_points, centx, centy)

plt.figure(figsize=(8,8))
plt.imshow(img, alpha=0.2)
plt.scatter(hull_points.T[0],hull_points.T[1],s=10,c='r',marker='+')
plt.scatter(centx,centy,marker='x',s=80,c='r')

idx = 0
for p in range(len(spiral_points)-1):
    idx += 1
    plt.plot([spiral_points[p][0],spiral_points[p+1][0]],[spiral_points[p][1],spiral_points[p+1][1]],c='royalblue')
    plt.text(spiral_points[p][0],spiral_points[p][1],str(idx),fontsize=6)
plt.show()

# printer/gcode parameters
#printer='prusa_i3' # generic / ultimaker2plus / prusa_i3 / ender_3 / cr_10 / bambulab_x1 / toolchanger_T0
gcode_filename = 'failure_patch'

z_height = 25.7
layer_height = 0.2
img_to_real_scale = 17.1
img_to_real_origin = (105.2,100.3) # xy

gcode_controls = fc.GcodeControls(
    printer_name = 'prusa_i3',
    initialization_data={
    'primer': 'front_lines_then_y',
    'print_speed': 45,
    "nozzle_temp": 210,
    "bed_temp": 60,
    "extrusion_width": 0.4,
    "extrusion_height": z_height,
    "fan_percent": 50})

steps = generate_gcode_steps(spiral_points, z_height, layer_height, img_to_real_scale, img_to_real_origin)
gcode = fc.transform(steps, 'gcode', gcode_controls)

fc.transform(steps, 'plot', fc.PlotControls(neat_for_publishing=False, zoom=1))
print(gcode[:500])

open(f'{gcode_filename}.gcode', 'w').write(gcode)
from google.colab import files
files.download(f'{gcode_filename}.gcode')


