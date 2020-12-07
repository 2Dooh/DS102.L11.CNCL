import numpy as np

def convert_coords(coord, length):
    # 0 : xmin, 1: xmax, 2:ymin, 3:ymax
    # [501, 511, 481, 503],  -> lay 4 so chia 608
    # 0.82, 0.84, 0.83, 0.87
    # converted_coords = 
    # 0 (xmin + xmax)/2: x trung diem
    #   (ymin + ymax)/2: y trung diem 
    #   x_trungdiem - xmin: chenh lech x
    #   y_trungdiem - ymin: chenh lech y

    coord = np.array(coord)
    coord = coord/length
    xmin, xmax, ymin, ymax = coord
    x_midpoint = (xmin + xmax)/2
    y_midpoint = (ymin + ymax)/2
    x_midrange = x_midpoint - xmin
    y_midrange = y_midpoint - ymin
    converted_coord = [x_midpoint, y_midpoint, x_midrange, y_midrange]
    return converted_coord

def reconvert_coords(coord, dict):
    label_number, x_midpoint, y_midpoint, x_midrange, y_midrange = coord
    label_name = (list(dict.keys())[list(dict.values()).index(label_number)])
    xmin = round((x_midpoint - x_midrange)*608)
    xmax = round((x_midpoint + x_midrange)*608)
    ymin = round((y_midpoint - y_midrange)*608)
    ymax = round((y_midpoint + y_midrange)*608)
    reconverted_coord = [xmin, xmax, ymin, ymax]
    return reconverted_coord, label_name