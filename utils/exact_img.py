import cv2 as cv
import json
import xmltodict
import matplotlib.pyplot as plt
import os
annot_path = '../data/labeled_images'
img_path = '../img'
# i = 0
annotations = os.listdir(annot_path)
for annot in annotations:
    with open(os.path.join(annot_path, annot)) as f:
        annot_dict = xmltodict.parse(f.read()) #read giong doan nay ne`
        # print(annot_dict)
        json_data = json.dumps(annot_dict)
    with open('test.json', 'w+') as writer:
        writer.write(json_data)
        writer.close()
        with open('test.json', 'r+') as file:
            annotation = json.load(file) 
            # for i in len(annotation['annotation']['object']):
            object = annotation['annotation']['object']
            for i in len(object):
                
            label = object['name']
            coordinate = object['bndbox']
            xmin, ymin, xmax, ymax = [int(coord) for coord in coordinate.values()] 
            img = cv.imread(img_path)
            cropped_img = img[ymin:ymax, xmin:xmax]
            save_path = label
            if (not os.path.exists(save_path)):
                os.makedirs(save_path)
            cv.imwrite(os.path.join(save_path, '.jpg'.format(0)), cropped_img)
        
        # annotation = json_data['ann']
  
    # #     print(annot)
    # # i+=1
    # #     # json_file = 'test.json'
    #     with open(json_file, 'w+') as file:
    #     annotation = json.load(file)
    #     object = annotation['annotation']['object']
    #     label = object['name']
    #     coordinate = object['bndbox']

    #     xmin, ymin, xmax, ymax = [int(coord) for coord in coordinate.values()] 

    #     img = cv.imread(img_path)
    #     cropped_img = img[ymin:ymax, xmin:xmax]

    #     save_path = label

    #     if (not os.path.exists(save_path)):
    #         os.makedirs(save_path)

    #     cv.imwrite(os.path.join(save_path, '.jpg'.format(0)), cropped_img)

    