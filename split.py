import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import rotate
from scipy.signal import argrelmin

warnings.simplefilter('ignore')
class TextProcessor:
    def __init__(self):
        pass

    def createkernel(self, kernel_size, sigma, theta):
        """Create anisotropic filter kernel according to given parameters"""
        assert kernel_size % 2  # must be odd size
        half_size = kernel_size // 2

        kernel = np.zeros([kernel_size, kernel_size])
        sigma_x = sigma
        sigma_y = sigma * theta

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - half_size
                y = j - half_size

                exp_term = np.exp(-x**2 / (2 * sigma_x) - y**2 / (2 * sigma_y))
                x_term = (x**2 - sigma_x**2) / (2 * math.pi * sigma_x**5 * sigma_y)
                y_term = (y**2 - sigma_y**2) / (2 * math.pi * sigma_y**5 * sigma_x)

                kernel[i, j] = (x_term + y_term) * exp_term

        kernel = kernel / np.sum(kernel)
        return kernel

    def applysummfunction(self, img):
        """Sum elements in columns"""
        res = np.sum(img, axis=0)
        return res

    
    def normalize(self, img):
        (m, s) = cv2.meanStdDev(img)
        m = m[0][0]
        s = s[0][0]
        img = img - m
        img = img / s if s > 0 else img
        return img

    
    def smooth(self, x, window_len=11, window='hanning'):
    #     if x.ndim != 1:
    #         raise ValueError("smooth only accepts 1 dimension arrays.") 
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.") 
        if window_len<3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") 
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(),s,mode='valid')
        return y

    def croptexttolines(self, text, blanks):
        x1 = 0
        y = 0
        lines = []
        for i, blank in enumerate(blanks):
            x2 = blank
            #print("x1=", x1, ", x2=", x2, ", Diff= ", x2-x1)
            line = text[:, x1:x2]
            lines.append(line)
            x1 = blank
        return lines

    def transposelines(self,lines):
        res = []
        for l in lines:
            line = np.transpose(l)
            img = rotate(line, 180)
            flip = np.flipud(img)
            res.append(flip)
        return res

    def splitimg(self,imgdir):
        img1 = imgdir
        img2 = img1[int(img1.shape[0]*0.4):img1.shape[0], int(0.3*img1.shape[1]):img1.shape[1]] #cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        imgn = np.transpose(img2)
        img3 = np.flipud(imgn)
        #img = np.arange(16).reshape((4,4))
        kernelSize=9
        sigma=4
        theta=1.5
        imgFiltered1 = cv2.filter2D(img3, -1, self.createkernel(kernelSize, sigma, theta), borderType=cv2.BORDER_REPLICATE)
        img4 = self.normalize(imgFiltered1)
        summ = self.applysummfunction(img4)
        smoothed = self.smooth(summ, 72) 
        mins = argrelmin(smoothed, order=2)
        arr_mins = np.array(mins)
        found_lines = self.croptexttolines(img3, arr_mins[0])
        sess = tf.compat.v1.Session()
        found_lines_arr = []
        with sess.as_default():
            for i in range(len(found_lines)-1):
                found_lines_arr.append(tf.expand_dims(found_lines[i], -1).numpy())
        res_lines = self.transposelines(found_lines)
        return res_lines



