from numpy.core.fromnumeric import shape
from .resize_img import square_padding_img
import cv2 as cv
import json
import xmltodict
import os
import click
import re
import numpy as np
import matplotlib.pyplot as plt
from .convert_coord import convert_coords, reconvert_coords



@click.command()
@click.option('--annot_path', required=True)
@click.option('--img_path', required=True)
@click.option('--save_path', default='./data/cropped_imgs')
@click.option('--extension', '-ext', default='jpg')
@click.option('--resize', type=(int, int), required=True)
@click.option('--interpolation', '-inter', default='INTER_AREA')
def cli(annot_path, 
        img_path, 
        save_path, 
        extension, 
        resize,
        interpolation):  
    for idx_i, annot in enumerate([f for f in os.listdir(annot_path) if f.endswith('.xml')]):
        filename = re.sub(r'\..+', '', annot)
        with open(os.path.join(annot_path, annot)) as handle:
            ordered_dict = xmltodict.parse(handle.read())
        annotation = json.loads(json.dumps(ordered_dict)) 
        objects = annotation['annotation']['object']
        # remove label bicycle
        objects = [i for i in objects if not (i['name'] == 'bicycle')]
        for idx_j, obj in enumerate(objects):
            try:
                label, coords = obj['name'], obj['bndbox']
                coord = np.array(list(coords.values()), dtype=np.float).reshape((2, 2)).T

                img = cv.imread(os.path.join(img_path, filename + '.{}'.format(extension)))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # scale = max(resize) / max(img.shape[:-1])

            # padded_img, bias = square_padding_img(img)
            # resized_img = cv.resize(padded_img, resize, interpolation=getattr(cv, interpolation))

            # coord *= scale
            # coord[np.argmax(img.shape[:-1])] += scale * bias
            
                xmin, xmax, ymin, ymax = coord.ravel().astype(np.int)
                w, h = xmax-xmin, ymax-ymin
                center_x, center_y = xmin + w//2, ymin + h//2
                bias = max(h, w) // 2
                xmin, xmax = center_x - bias, center_x + bias
                ymin, ymax = center_y - bias, center_y + bias
                padded_cropped_obj = img[ymin:ymax, xmin:xmax]
                
                # padded_cropped_obj, _ = square_padding_img(cropped_obj, mode='edge')
                interpolation = 'INTER_AREA' if max(resize) < max(padded_cropped_obj.shape) else 'INTER_CUBIC'
                resized_img = cv.resize(padded_cropped_obj, 
                                        resize, 
                                        interpolation=getattr(cv, interpolation))
                class_folder = save_path + '/{}'.format(label)
                if (not os.path.exists(class_folder)):
                    os.makedirs(class_folder)
                cv.imwrite(os.path.join(class_folder, '{}_{}.{}'.format(idx_i, idx_j, extension)), resized_img)
                # plt.imshow(padded_cropped_obj)
                # plt.show()
            except:
                pass
            
            
if __name__ == '__main__':
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608, 608])
    # cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/rgb_32x32', '--resize', 32, 32])
    pass
    # cli()