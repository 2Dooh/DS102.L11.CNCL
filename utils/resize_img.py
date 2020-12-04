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
        resized_img = cv.resize(padded_img, (resize, resize), interpolation=getattr(cv, interpolation))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv.imwrite(os.path.join(save_path, img_name), resized_img)
        

def square_padding_img(img, mode='edge'):
    bias = (np.abs(np.diff(img.shape[:-1])) // 2).astype(np.int)[0]
    n_pads = ((0, 0), (bias, bias), (0, 0)) if img.shape[0] > img.shape[1] else ((bias, bias), (0, 0), (0, 0))
    padded_img = np.pad(img, n_pads, mode=mode, constant_values=0)
    # assert(padded_img.shape[0] == padded_img.shape[1])
    return padded_img, bias
            
            
if __name__ == '__main__':
    # resize_img_cli(['--annot_path', './data/labeled_images/', '--img_path', './data/video20-305/', '--save_path',
    #     './data/CroppedImages608x608', '--resize', 608])
    resize_img_cli(['--img_path', './data/video20-305', '--resize', 608])