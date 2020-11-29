import cv2 as cv
import json
import xmltodict
import os
import click
import re

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
        
    for idx_i, annot in enumerate(os.listdir(annot_path)):
        filename = re.sub(r'\..+', '', annot)

        with open(os.path.join(annot_path, annot)) as handle:
            ordered_dict = xmltodict.parse(handle.read())
        annotation = json.loads(json.dumps(ordered_dict)) 
        objects = annotation['annotation']['object']
        for idx_j, obj in enumerate(objects):
            label, coords = obj['name'], obj['bndbox']
            [xmin, ymin, xmax, ymax] = [int(coord) for coord in coords.values()]

            img = cv.imread(os.path.join(img_path, filename + '.{}'.format(extension)))
            cropped_obj = img[ymin:ymax, xmin:xmax]
            resized_img = cv.resize(cropped_obj, 
                                    resize, 
                                    interpolation=getattr(cv, interpolation))

            class_folder = save_path + '/{}'.format(label)
            if (not os.path.exists(class_folder)):
                os.makedirs(class_folder)
            cv.imwrite(os.path.join(class_folder, '{}_{}.{}'.format(idx_i, idx_j, extension)), resized_img)
            
if __name__ == '__main__':
    cli()