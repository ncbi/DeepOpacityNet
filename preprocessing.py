import numpy as np

import cv2

import matplotlib.pyplot as plt



def normalize_img(img):
    
    return (img / 127.5) - 1.0


def hist_equalize(img):
    
    red = img[:,:,0]

    green = img[:,:,1]

    blue = img[:,:,2]

    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))

    red_eq = clahe.apply(red)

    green_eq = clahe.apply(green)

    blue_eq = clahe.apply(blue)

    img = np.stack([red_eq, green_eq, blue_eq], axis=-1)


    return img
    

def preprocess_img(imgpath):
    
    img = cv2.imread(imgpath)
            
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
        
    img = hist_equalize(img)
    
    img = normalize_img(img)
    
    return img


def match_dims(img):
        
    img = np.expand_dims(img, axis=0)
        
    return img

def draw_img(img):
    
    plt.figure()
    
    plt.imshow(img)
    
    plt.axis('off')
             
    plt.show()