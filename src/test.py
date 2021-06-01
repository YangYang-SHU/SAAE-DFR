import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

path = '/mnt/ssd/dk-projects/multifeature/mvtec_data_bak/'

def get_image_files(path, mode='train'):
    images = []
    ext = {'.jpg', '.png'}
#     path = "/home/jie/Datasets/mvtec-anomaly/bottle/test"
    for root, dirs, files in os.walk(path):
        # print('loading image files ' + root)
        for file in files:
            if mode == 'train':
                if os.path.splitext(file)[1] in ext:
                    images.append(os.path.join(root, file))
            else:
                if os.path.splitext(file)[1] in ext and "good" not in root:
                    images.append(os.path.join(root, file))
    return sorted(images)


def get_mask_files(path):
    masks = []
    ext = {'.jpg', '.png'}
#     path = "/home/jie/Datasets/mvtec-anomaly/bottle/ground_truth"
    for root, dirs, files in os.walk(path):
        # print('loading mask files ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                masks.append(os.path.join(root, file))
    return sorted(masks)

if __name__ == "__main__":
    textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
    objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    data_names = objects + textures
    for dn in data_names:
        paths = path + dn
        print(paths)
        imgs = get_image_files(paths)
        for pp in imgs:
            print(pp)
            img = cv2.imread(pp)
            new_dir = pp.rsplit("/",1)[0]
            newpp = pp.replace('mvtec_data_bak','mvtec_anomaly_detec')
            img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
            print(img.shape)
            if not os.path.exists(new_dir):
                print(new_dir)
                os.makedirs(new_dir)
            print(newpp)
            cv2.imwrite(newpp, img)  # 保存图像
    
        
    

