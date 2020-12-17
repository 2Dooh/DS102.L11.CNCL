from typing import Any
import cv2 as cv
import json
import xmltodict
import os
import click
import re
import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option('--img_path', required=True)
@click.option('--save_path', default='./data/resized_imgs')
@click.option('--extension', '-ext', default='jpg')
@click.option('--resize', type=int, required=True)
@click.option('--interpolation', '-inter', default='INTER_AREA')
def resize_img_cli( 
        img_path, 
        save_path, 
        extension, 
        resize,
        interpolation):  
    
    for idx_i, img_name in enumerate(os.listdir(img_path)):
        # filename = re.sub(r'\..+', '', img_name)

        img = cv.imread(os.path.join(img_path, img_name))
        padded_img, _ = square_padding_img(img, mode='constant')
        resized_img = cv.resize(padded_img, 
                                (resize, resize), 
                                interpolation=getattr(cv, interpolation))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv.imwrite(os.path.join(save_path, img_name), resized_img)


@click.command()
@click.option('--img_dir', required=True)
@click.option('--target_dir', default='./data/resized_imgs')
@click.option('--resize', type=int, required=True)
@click.option('--mode', '-m', default='constant')
def resize_img_folder(img_dir, 
                      target_dir, 
                      mode, 
                      resize):
    # os.makedirs(target_dir, exist_ok=True)
    classes = os.listdir(img_dir)
    for cls in classes:
        target_dir_class = os.path.join(target_dir, cls)
        os.makedirs(target_dir_class, exist_ok=True)

        class_path = os.path.join(img_dir, cls)
        image_names = os.listdir(class_path)
        for img_name in image_names:
            try:
                img_path = os.path.join(class_path, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                padded_img, _ = square_padding_img(img, mode=mode)
                interpolation = 'INTER_AREA' if resize < max(padded_img.shape) else 'INTER_CUBIC'
                resized_img = cv.resize(img, (resize, resize), interpolation=getattr(cv, interpolation))

                cv.imwrite(os.path.join(target_dir_class, img_name), resized_img)
            except:
                pass

        

def square_padding_img(img, mode='edge'):
    bias = (np.abs(np.diff(img.shape[:-1])) // 2).astype(np.int)[0]
    n_pads = ((0, 0), (bias, bias), (0, 0)) if img.shape[0] > img.shape[1] else ((bias, bias), (0, 0), (0, 0))
    padded_img = np.pad(img, n_pads, mode=mode)
    return padded_img, bias

class SquarePadding(object):
    def __init__(self, mode='constant') -> None:
        super().__init__()
        self.mode = mode

    def __call__(self, img: np.ndarray) -> np.ndarray:
        bias = (np.abs(np.diff(img.shape[:-1])) // 2).astype(np.int)[0]
        n_pads = ((0, 0), (bias, bias), (0, 0)) \
            if img.shape[0] > img.shape[1] else ((bias, bias), (0, 0), (0, 0))
        padded_img = np.pad(img, n_pads, mode=self.mode)
        return padded_img

            
            
if __name__ == '__main__':
    # resize_img_cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608])
    # resize_img_cli(['--img_path', './data/video20-305', '--resize', 608])
    # resize_img_folder(['--img_dir', 
    #                    './data/UIT-VC/Test/', 
    #                    '--target_dir', 
    #                    './data/UIT-VC-constant-padded/Test', 
    #                    '--resize', 
    #                    '32',
    #                    '--mode',
    #                    'constant'])
    pass