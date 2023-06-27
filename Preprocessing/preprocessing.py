import numpy as np

import cv2

from sklearn.mixture import GaussianMixture

from scipy import stats

import matplotlib.pyplot as plt

def get_shapes(img_gray):

    try:
        
        H, W = img_gray.shape # 0:H, 1:W
        
        
        img_blur = cv2.GaussianBlur(img_gray, (5,5), cv2.BORDER_DEFAULT) # img_gray.copy()

        cutoff = 5

        data = img_blur.ravel()

        data = data[data >= cutoff]

        ## fit GMM

        gmm = GaussianMixture(n_components = 3, n_init=5, random_state=1) # 

        gmm = gmm.fit(X=np.expand_dims(data,1))

        ## evaluate GMM

        gmm_x = np.arange(0, 256) 

        gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))

        gmm_y_cum = np.cumsum(gmm_y)

        means_hat = gmm.means_.flatten()

        sds_hat = np.sqrt(gmm.covariances_).flatten()

        weights_hat = gmm.weights_.flatten()

        min_id = np.argmin(means_hat)

        # sorting means of the distributions

        idx = np.argsort(means_hat)

        # weighted distributions

        dst0 = weights_hat[idx[0]] * stats.norm.pdf(gmm_x, means_hat[idx[0]], sds_hat[idx[0]])

        dst1 = weights_hat[idx[1]] * stats.norm.pdf(gmm_x, means_hat[idx[1]], sds_hat[idx[1]])

        dst2 = weights_hat[idx[2]] * stats.norm.pdf(gmm_x, means_hat[idx[2]], sds_hat[idx[2]])


        ## area of each distribution ... the sum of the probabilities

        area0 = np.sum(dst0)

        area1 = np.sum(dst1)

        area2 = np.sum(dst2)


        ## relative areas to the total area

        total_area = area0 + area1 + area2

        area0_r = area0 / total_area

        area0_r = np.round(area0_r, 4)

        area1_r = area1 / total_area

        area1_r = np.round(area1_r, 4)

        area2_r = area2 / total_area

        area2_r = np.round(area2_r, 4)



        ## the intersection of each distribtuion with the 1st distribution ... the minimum

        min01 = np.minimum(dst0, dst1)

        min02 = np.minimum(dst0, dst2)


        ## relative intersection areas to the area of the union

        # intersection areas

        inter01 = np.sum(min01)

        inter02 = np.sum(min02)

        # relative intersection using dice coefficient

        inter01_r = (2 * inter01) / (area0 + area1) # 1st and 2nd

        inter01_r = np.round(inter01_r, 4)


        inter02_r = (2 * inter02) / (area0 + area2) # 1st and 3rd

        inter02_r = np.round(inter02_r, 4)


        inter_r = 2 * (inter01 + inter02) / total_area # 1st and 2nd + 3rd

        inter_r = np.round(inter_r, 4)


        ## means and sds 

        min_mean = means_hat[idx[0]]

        min_sd = sds_hat[idx[0]]    

        med_mean = means_hat[idx[1]]

        med_sd = sds_hat[idx[1]]

        max_mean = means_hat[idx[2]]

        max_sd = sds_hat[idx[2]]


        # calculating thresholds between disributions

        thresh0 = min_mean + 3 * min_sd

        sd_thresh = 32

        area_thresh = 0.05

        inter_thresh = 0.01

        if (area1_r <= area_thresh) and (inter01_r <= inter_thresh): # negligible 2nd distribution 

            thresh1 = med_mean + 3 * med_sd

        elif (med_sd >= sd_thresh): # negligible 2nd distribution

            thresh1 = med_mean + 3 * med_sd

        elif (med_mean - 3 * med_sd <= 0): # negligible 2nd distribution

            thresh1 = med_mean + 3 * med_sd

        else: # eligible 2nd distribution

            thresh1 = med_mean - 3 * med_sd


        if (area2_r <= area_thresh) and (inter02_r <= inter_thresh): # negligible 3rd distribution

            thresh2 = max_mean + 3 * max_sd

        elif (max_sd >= sd_thresh) : # negligible 3rd distribution

            thresh2 = max_mean + 3 * max_sd


        elif (max_mean - 3 * max_sd <= 0): # negligible 3rd distribution

            thresh2 = max_mean + 3 * max_sd

        else: # eligible third distribution

            thresh2 = max_mean - 3 * max_sd



        ## the nearest threshold to the 1st distribution

        thresh12 = min([thresh1, thresh2])


        ## the cutoff corresponding to 1.5%

        position = len(gmm_y[gmm_y_cum<=0.015])

        cutoff_1p5 = gmm_x[position] # cutoff 1 point 5

        # larger intersection or larger-area 1st distribution
        if inter_r >= 0.1 or area0_r >= 0.4: 

            img_thresh = cutoff_1p5

        # separated 1st and 2nd/3rd    
        elif thresh0 <= thresh12 and min_sd <=15: 

            img_thresh = (thresh0 + thresh12)/2

        # separated 1st and 2nd/3rd w/ small overlap                                
        elif thresh0 > thresh12 and thresh12 > min_mean and min_sd <=10: 

                img_thresh = thresh12

        # small intersection w/ small sd for 1st distibution            
        elif inter_r <= 0.01 and min_sd <=10: 

            img_thresh = np.round(min_mean + 3 * min_sd)

        # small intersection w/ intermediate sd for 1st distibution
        elif inter_r <= 0.01 and min_sd <=20:

            img_thresh = np.round(min_mean + 1 * min_sd)

        # small intersection for 1st distibution
        elif inter_r <= 0.01:

            img_thresh = np.round(min_mean)

        # any other case ....
        else:

            img_thresh = cutoff_1p5


        img_thresh = int(img_thresh)



        ## converting the gray img into binary image

        (_, bw_img) = cv2.threshold(img_gray, img_thresh, 255, cv2.THRESH_BINARY) # img_thresh



        thresh = bw_img.copy()


        ## removing any small comp using OPEN operation

        element = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(75, 75))

        thresh = cv2.morphologyEx(thresh, op=cv2.MORPH_OPEN, kernel=element)

        # if np.sum(thresh) == 0: # black image

        #    thresh = bw_img.copy() # copy the binary image again ... remove morphological operations ... 


    
        ## finding contours

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        ## sorting comp using their areas

        areas = [cv2.contourArea(c) for c in contours]

        sorted_areas = np.sort(areas)


        ## contour

        cnt = contours[areas.index(sorted_areas[-1])] # the biggest contour

        ## bounding rectangle

        Rect = cv2.boundingRect(cnt)
                        
        
        ## min circle

        (xc, yc), radius = cv2.minEnclosingCircle(cnt)
        
        circle = (xc/W, yc/H, radius/min(W, H))


        ## max inscribed square

        # has the same center, but w/ side = sqrt(2) * r,
        # the distance between center to any side = r / sqrt(2)

        xs = xc - radius / np.sqrt(2)

        ys = yc - radius / np.sqrt(2)


        ws = np.sqrt(2) * radius

        hs = np.sqrt(2) * radius


        inner_bb = (xs/W, ys/H, ws/W, hs/H)
        
        ## adjusted iscribed square

        if xs < 0:

            ws = ws - np.abs(xs)

            xs = 0        

        if ys < 0:

            hs = hs - np.abs(ys)

            ys = 0

        if xs + ws > W:

            ws = W - xs

        if ys + hs > H:

            hs = H - ys
                
        
        inner_bb_adj = (xs/W, ys/H, ws/W, hs/H)


        ## circumscribed square

        # has the same center, and w/ side = 2*r,

        xs = xc - radius

        ys = yc - radius


        ws = 2 * radius

        hs = 2 * radius


        outer_bb = (xs/W, ys/H, ws/W, hs/H)
        
        ## adjusted circumscribed square

        if xs < 0:

            ws = ws - np.abs(xs)

            xs = 0        

        if ys < 0:

            hs = hs - np.abs(ys)

            ys = 0

        if xs + ws > W:

            ws = W - xs

        if ys + hs > H:

            hs = H - ys
        
        
        outer_bb_adj = (xs/W, ys/H, ws/W, hs/H)
        

    except Exception as e:               
        
        circle = (0.5, 0.5, 0.5)
               
        inner_bb = (0, 0, 1, 1)
        
        outer_bb = (0, 0, 1, 1)
                               
        inner_bb_adj = (0, 0, 1, 1)
        
        outer_bb_adj = (0, 0, 1, 1)
    
    return circle, inner_bb, inner_bb_adj, outer_bb, outer_bb_adj
    
