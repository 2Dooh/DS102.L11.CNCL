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
from utils.convert_coord import convert_coords, reconvert_coords



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
        # remove label bicycle
        objects = [i for i in objects if not (i['name'] == 'bicycle')]
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



    # f.close()        
if __name__ == '__main__':
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608, 608])
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608, 608])
    cli()