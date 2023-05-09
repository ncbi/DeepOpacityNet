"""

Detects cataract in dataset.

Usage:

    classify.py --model_folder=<str> --image_folder=<str> --output_file=<str>

Options:
    --model_folder=<str>       the path of the models
    --image_folder=<str>        the path of the images
    --output_file=<str>        the output file
"""


import sys 

import os

from docopt import docopt

import numpy as np

import pandas as pd

import preprocessing

import trained_model


def classify_main(model_folder='Model', img_folder='CFP', output='predictions.csv'):
            
    print('Loading DeepOpacityNet')

    model_path = os.path.join(model_folder, 'DeepOpacityNet.h5')

    model = trained_model.load(model_path)
        
    # model.summary()
        
    img_list = [img for img in os.listdir(img_folder) if img.endswith('.png')]
    
    print('#imgs in folder is', len(img_list))
    
    predictions = []
    
    for filename in img_list:
        
        print('predicting img', filename)
        
        filepath = os.path.join(img_folder, filename)
        
        img = preprocessing.preprocess_img(filepath)
        
        # print(img.shape)
        
        # preprocessing.draw_img(img)
        
        img = preprocessing.match_dims(img)
        
        pred = model.predict(img, verbose=0)
        
        print(pred[0])
            
        predictions.append(pred[0])
    
    
    predictions = np.array(predictions)
    
    df = pd.DataFrame(predictions)
    
    df.to_csv(output, index=False)
    
            

if __name__ == "__main__":
        
    argv = docopt(__doc__, sys.argv[1:])
    
    if len(argv) == 3: # parameters are passed
    
        model_folder = argv['--model_folder']

        img_folder = argv['--image_folder']

        output = argv['--output_file']

        classify_main(model_folder, image_folder, output)
    
    else: # no parameters, using the defaults
        
        classify_main()
    





















