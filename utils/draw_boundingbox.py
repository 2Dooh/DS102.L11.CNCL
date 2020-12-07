import cv2 as cv
import matplotlib.pyplot as plt
def draw_boundingbox(img_path, txt_path):
    img = cv.imread(img_path)
    dh, dw, _ = img.shape

    fl = open(txt_path, 'r')
    data = fl.readlines()
    fl.close()

    target_dict = {
                    "bus": 0,
                    "car": 1,
                    "motor": 2,
                    "others": 3,
                    "pedestrian": 4,
                    "truck": 5,
                    "van": 6
                }

    for dt in data:

        label_number, x, y, w, h = map(float, dt.split(' '))

        label_name = (list(target_dict.keys())[list(target_dict.values()).index(label_number)])
        l = int((x - w) * dw)
        r = int((x + w) * dw)
        t = int((y - h) * dh)
        b = int((y + h) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv.rectangle(img, (l, t), (r, b), (255, 0, 0), 2)
        cv.putText(img, label_name, (int(l) , int(t) - 3), cv.FONT_HERSHEY_SIMPLEX , 0.3, (255, 0, 0))

    plt.imshow(img)

    plt.savefig('./data/test.png')
    # plt.show()
