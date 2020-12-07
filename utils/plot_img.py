import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle

def main():
    names = ['bus', 'car', 'motor', 'background', 'pedestrian', 'truck', 'van']
    colors = [(88, 235, 52), 
            (88, 235, 52), 
            (235, 235, 52), 
            (229, 52, 235), 
            (36, 209, 198), 
            (209, 36, 62), 
            (172, 209, 36)]
    img_path = './data/video20-305/001.jpg'
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # img = cv.resize(img, (1280, 720))
    with open('./pred/pred_1_fhd.pickle', 'rb') as handle:
        res = pickle.load(handle)
    boxes, labels, scores = res.values()
    stride = 4
    for coord, label, score in zip(boxes, labels, scores):
        if score > 0.9:
            start_point = (coord[::-1] * img.shape[:-1][::-1] * stride).astype(np.int)
            end_point = start_point + 32
            img = cv.rectangle(img, 
                            tuple(start_point.tolist()), 
                            tuple(end_point.tolist()), 
                            color=colors[label], 
                            thickness=2)
            c1, c2 = start_point
            img = cv.putText(img, '{}'.format(names[label]), 
                            (c1, c2-10), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            0.5, colors[label], 2)
    plt.imshow(img)
    # plt.imsave('./001.jpg', img)
    plt.show()
    # print(res.shape)
    #print(np.unique(res, return_counts=True))

# main()