import cv2 as cv
import json
import xmltodict
import os
import click
import re
import numpy as np

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
    
    set_size_img = 608
    for idx_i, annot in enumerate(os.listdir(annot_path)):
        filename = re.sub(r'\..+', '', annot)
        with open(os.path.join(annot_path, annot)) as handle:
            ordered_dict = xmltodict.parse(handle.read())
        annotation = json.loads(json.dumps(ordered_dict)) 
        objects = annotation['annotation']['object']
        for idx_j, obj in enumerate(objects):
            label, coords = obj['name'], obj['bndbox']
            coordinate = [int(coord) for coord in coords.values()] 
            coord = np.array(coordinate)
            img = cv.imread(os.path.join(img_path, filename + '.{}'.format(extension)))
            h, w, c = img.shape # height, width, color 
            if w > h: 
                img_scale = set_size_img / w
                img_v_pad = np.zeros([int((w-h)/2), w, c])
                img = np.vstack([img_v_pad, img, img_v_pad])
                coord = coord*img_scale
                coord[1] += img_scale * ((w-h)/2)
                coord[3] += img_scale * ((w-h)/2)
            elif h > w:
                img_scale = set_size_img / h
                img_h_pad = np.zeros([h, int((h-w)/2), c])
                img = np.hstack([img_h_pad, img, img_h_pad])
                coord = coord*img_scale
                coord[0] += img_scale * ((h-w)/2)
                coord[2] += img_scale * ((h-w)/2)
            coord = coord.astype(int)
            [xmin, ymin, xmax, ymax] = coord
            # resize img 
            rs_img = cv.resize(img, (set_size_img, set_size_img))
            cropped_obj = rs_img[ymin:ymax, xmin:xmax]
            # resized_img = cv.resize(cropped_obj, 
            #                         resize, 
            #                         interpolation=getattr(cv, interpolation))
            class_folder = save_path + '/{}'.format(label)
            if (not os.path.exists(class_folder)):
                os.makedirs(class_folder)
            cv.imwrite(os.path.join(class_folder, '{}_{}.{}'.format(idx_i, idx_j, extension)), cropped_obj)
            
            
if __name__ == '__main__':
    cli()