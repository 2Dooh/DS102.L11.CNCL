from numpy.core.fromnumeric import shape
from resize_img import square_padding_img
import cv2 as cv
import json
import xmltodict
import os
import click
import re
import numpy as np
import matplotlib.pyplot as plt




@click.command()
@click.option('--annot_path', required=True)
@click.option('--img_path', required=True)
@click.option('--save_path', default='./data/cropped_imgs')
@click.option('--extension', '-ext', default='jpg')
@click.option('--resize', type=(int), required=True)
@click.option('--interpolation', '-inter', default='INTER_AREA')
def cli(annot_path, 
        img_path, 
        save_path, 
        extension, 
        resize,
        interpolation):  
    for idx_i, annot in enumerate(os.listdir(annot_path)):
        filename = re.sub(r'\..+', '', annot)
        with open(os.path.join(annot_path, annot)) as handle:
            ordered_dict = xmltodict.parse(handle.read())
        annotation = json.loads(json.dumps(ordered_dict)) 
        objects = annotation['annotation']['object']
        for idx_j, obj in enumerate(objects):
            label, coords = obj['name'], obj['bndbox']
            coord = np.array(list(coords.values()), dtype=np.float).reshape((2, 2)).T

            img = cv.imread(os.path.join(img_path, filename + '.{}'.format(extension)))
            scale = resize / max(img.shape[:-1])

            padded_img, bias = square_padding_img(img)
            resized_img = cv.resize(padded_img, (resize, resize), interpolation=getattr(cv, interpolation))

            coord *= scale
            coord[np.argmax(img.shape[:-1])] += scale * bias
            xmin, xmax, ymin, ymax = coord.ravel().astype(np.int)
            
            coord_data = [xmin, xmax, ymin, ymax]
            converted_coord = convert_coords(coord_data, resized_img.shape[0])

            target_dict = {
                "bus": 0,
                "car": 1,
                "motor": 2,
                "others": 3,
                "pedestrian": 4,
                "truck": 5,
                "van": 6
            }
            converted_coord.insert(0,target_dict[label])
            # print(str(converted_coord).replace(',', ''))
            # print(reconvert_coords(converted_coord, target_dict))
            # save to txt file

            # yolo_target = [label] + convert_coords(coord, img.shape[0])
            # print(reconvert_coords(yolo_target))

            # save coord_data
            

            # cropped_obj = img[ymin:ymax, xmin:xmax]

            # try:
            #     padded_cropped_obj, _ = square_padding_img(cropped_obj)
            #     interpolation = 'INTER_AREA' if max(resize) < max(padded_cropped_obj.shape) else 'INTER_CUBIC'
            #     resized_img = cv.resize(padded_cropped_obj, 
            #                             resize, 
            #                             interpolation=getattr(cv, interpolation))
            #     class_folder = save_path + '/{}'.format(label)
            #     if (not os.path.exists(class_folder)):
            #         os.makedirs(class_folder)
            #     cv.imwrite(os.path.join(class_folder, '{}_{}.{}'.format(idx_i, idx_j, extension)), resized_img)
            # except:
            #     pass
            file_txt = open("./data/yolo_txt/file"+str(idx_i)+".txt","a+")
            file_txt.write(str(converted_coord).replace(',', '')[1:-1] + '\n')
            file_txt.close()
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


    # f.close()        
if __name__ == '__main__':
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608, 608])
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608, 608])
    cli()