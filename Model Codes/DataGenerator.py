from tensorflow.keras.utils import Sequence, to_categorical

import numpy as np

import random

import os

import cv2

import pandas as pd

import preprocessing

import augmentation

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt


class DataGenerator(Sequence):

    def __init__(self, filepath='.', batch_size=32, target_size=(512, 512), shuffle=True, aug=True, scale='zero-mean',
                 hist_eq=True, attribute='CATLabels', full_img=False):

        
#         print(batch_size, target_size, shuffle, aug, scale)
        
        _df = pd.read_csv(filepath)
        
        paths = _df['filepath'].to_list()
        
        paths = [path.replace('/projects/', '/projects/ophthalmology_projects/') for path in paths]
        
        if full_img:
        
            paths = [path.replace('inscribed', 'circum') for path in paths]            
            
            
#             print(paths[0])
        
        paths = np.array(paths)
                
        
        # labels
        
        labels = _df[attribute].to_numpy() # CATLabels # SeverityCats

        nClasses = len(np.unique(labels))

        class_weight = compute_class_weight('balanced', np.unique(labels), labels)
        
        class_weight = dict(enumerate(class_weight))
        
        self.class_weight = class_weight
        
        self.paths = paths
        
        self.labels = labels
        
        self.target_size = target_size

        self.nClasses = nClasses

        self.sequence_length = np.int32(np.ceil(len(labels) / batch_size))

        self.batch_size = batch_size
        
#         self.epoch = 0
        
        self.shuffle = shuffle

        self.aug = aug
        
        self.shuffle_data()
        
        self.scale = scale
        
        self.hist_eq = hist_eq
        
#         print('\nepoch #{} - batch size {} - len(gen) {}\n'.format(self.epoch, self.batch_size, self.sequence_length))
    
            
    
    def shuffle_data(self):
        
        if self.shuffle:
                        
            grouped = list(zip(self.paths, self.labels))
        
            random.shuffle(grouped)
        
            paths, labels = zip(*grouped)
            
            self.paths = np.array(paths)
            
            self.labels = np.array(labels)


    def on_epoch_end(self):

        self.shuffle_data()
        
#         self.epoch = self.epoch + 1
        
#       if (self.epoch%10 == 0) and (self.batch_size < 50):

#             self.batch_size = self.batch_size + 5
            
#         self.batch_size = self.batch_size + 5
        
#         self.sequence_length = np.int32(np.ceil(len(self.labels) / self.batch_size))

#         print('\nepoch #{} - batch size {} - len(gen) {}\n'.format(self.epoch, self.batch_size, self.sequence_length))


    def __len__(self):

        return self.sequence_length
    
    
    def augment(self, img):
                
        if self.aug:            
                
#             if np.random.uniform(0, 1) >= 0.5:
                
#                 img = self.random_cropping(img)
            
            if np.random.uniform(0, 1) >= 0.5:
                
                img = cv2.flip(img, 1) # horizontal flip
                
            if np.random.uniform(0, 1) >= 0.5:
                
                img = cv2.flip(img, 0) # vertical flip            
                
            if np.random.uniform(0, 1) >= 0.5:
                
                img = np.transpose(img, axes=(1, 0, 2)) # transpose                
            
            if np.random.uniform(0, 1) >= 0.5:
                
                img = self.random_erase(img)
                
        
        return img

                
            
    def hist_equalize(self, img):
        
        if self.hist_eq:   
            
            red = img[:,:,0]

            green = img[:,:,1]

            blue = img[:,:,2]

            clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))

            red_eq = clahe.apply(red)

            green_eq = clahe.apply(green)

            blue_eq = clahe.apply(blue)

            img = np.stack([red_eq, green_eq, blue_eq], axis=-1)


        return img

    
    def random_erase(self, img):
            
        H, W = img.shape[:-1]

        Cx = np.random.randint(W)

        Cy = np.random.randint(H)

        w = np.random.randint(round(W/8), round(W/6))

        h = np.random.randint(round(W/8), round(H/6))

        minX = max(Cx-w, 0)

        maxX = min(Cx+w, W)

        minY = max(Cy-h, 0)

        maxY = min(Cy+h, H)

        # rand_img = np.random.randint(0, 256, img.shape)
            
        
        img[minY:maxY, minX:maxX, :] = 0 # rand_img[minY:maxY, minX:maxX, :] # 0 
        
        return img
    
    
    def translate(self, img):

        nrows, ncols = img.shape[:-1]      

        Tx = (-1)**(np.random.choice([0, 1])) * np.random.randint(10, 25)

        Ty = (-1)**(np.random.choice([0, 1])) * np.random.randint(10, 25)

        Tmat = np.float32([ [1, 0, Tx], [0, 1, Ty] ])

        img = cv2.warpAffine(img, Tmat, (ncols, nrows))

        return img
            

    def __getitem__(self, key):

        nClasses = self.nClasses

        batch_size = self.batch_size

        target_size = self.target_size
        
        paths = self.paths
        
        labels = self.labels
               
        
        if key == self.sequence_length - 1:
            
            batch_paths = paths[key*batch_size:]
            
            batch_labels = labels[key*batch_size:]
                
        else:
            
            batch_paths = paths[key*batch_size:(key+1)*batch_size]
            
            batch_labels = labels[key*batch_size:(key+1)*batch_size]  
        
        batch_imgs = []                

        for imgpath in batch_paths:           
            
            img = cv2.imread(imgpath)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
            img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_NEAREST)                                    

            img_rgb = self.hist_equalize(img_rgb)
            
            img_rgb = self.augment(img_rgb)                        
                
            if self.scale == 'max':
                
                img_rgb = 1.0 * img_rgb / np.max(img_rgb) # 0 -> 1 
            
            elif self.scale == 'zero-mean':
                
                img_rgb = 2.0 * img_rgb / np.max(img_rgb) - 1 # -1 -> 1            
            
            batch_imgs.append(img_rgb)
            
        batch_imgs = np.array(batch_imgs)

        batch_labels = np.array(batch_labels)
    
        batch_labels = to_categorical(batch_labels, num_classes=nClasses)
            
        
        return batch_imgs, batch_labels
    
   