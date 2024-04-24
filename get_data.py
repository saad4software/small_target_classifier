import glob
import numpy as np
import cv2

def get_data_pairs(path, class_num=1, use_filter=True):
    files = glob.glob(path)
    train_data = []
    y_data = []

    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

    for file in files:
        file2 = f"{file[:-5]}2.png"
        frame1 = cv2.imread(file)
        frame2 = cv2.imread(file2)

        if frame1 is None or frame2 is None:
            continue

        if use_filter:
            frame1 = cv2.filter2D(frame1,-1,np.array(kernel))
            frame2 = cv2.filter2D(frame2,-1,np.array(kernel))

        train_data += [[frame1, frame2]]
        # train_data += [frame2]
        
        
    y_shape = np.array(train_data).shape[0]
    y_data = [class_num for x in range(y_shape)]

    return train_data, y_data


def get_data_single(path, class_num=1):
    files = glob.glob(path)
    train_data = []
    y_data = []

    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

    for file in files:
        # file2 = f"{file[:-5]}2.png"
        frame1 = cv2.imread(file)
        # frame2 = cv2.imread(file2)

        if frame1 is None:
            continue

        frame1 = cv2.filter2D(frame1,-1,np.array(kernel))
        # frame2 = cv2.filter2D(frame2,-1,np.array(kernel))

        train_data += [frame1]
        # train_data += [frame2]
        
        
    y_shape = np.array(train_data).shape[0]
    y_data = [class_num for x in range(y_shape)]

    return train_data, y_data


