import cv2 as cv
import json
import xmltodict
import matplotlib.pyplot as plt
import os

annot_path = '../data/labeled_images'
img_path = '../img'
annotations = os.listdir(annot_path)
for idx, annot in enumerate(annotations):
    filename = annot.replace('.xml', '')
    with open(os.path.join(annot_path, annot)) as f:
        annot_dict = xmltodict.parse(f.read()) #read giong doan nay ne`
        json_data = json.dumps(annot_dict)
        annotation = json.loads(json_data) 
        objects = annotation['annotation']['object']
        for j, dic in enumerate(objects):
            label = dic['name']
            coordinate = dic['bndbox']
            [xmin, ymin, xmax, ymax] = [int(coord) for coord in coordinate.values()]
            img = cv.imread(os.path.join(img_path, filename + '.jpg'))
            cropped_img = img[ymin:ymax, xmin:xmax]
            resized_img = cv.resize(cropped_img, (200, 152), interpolation = cv.INTER_AREA)
            save_path = '../data/imgs/{}'.format(label)
            if (not os.path.exists(save_path)):
                os.makedirs(save_path)
            cv.imwrite(os.path.join(save_path, '{}_{}.jpg'.format(idx, j)), resized_img)
            
            # label = [sub['name'] for sub in object]
            # label = object['name']
            # coordinate = [sub['bndbox'] for sub in object]
            # coordinate = object['bndbox']
            # xmin, ymin, xmax, ymax = [int(coord) for coord in coordinate.values()] 
            # img = cv.imread(img_path)
            # cropped_img = img[ymin:ymax, xmin:xmax]
            # save_path = label
            # if (not os.path.exists(save_path)):
            #     os.makedirs(save_path)
            # cv.imwrite(os.path.join(save_path, '.jpg'.format(0)), cropped_img)