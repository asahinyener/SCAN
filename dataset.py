#import configs.py as configs
import cv2
import os
import numpy as np
import random
def read_data(DATA_PATH,SAVE_PATH):
        image_paths=os.listdir(DATA_PATH)
        image_paths.sort(key=lambda x: int(x.replace("image","").replace(".png","")))
        hsv_images=[]
        for image_path in image_paths:
                image=cv2.imread(DATA_PATH+"/"+image_path)
                hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                hsv=np.transpose(hsv,(2,0,1))
                hsv_images.append(hsv)
        a=np.array(hsv_images)
        save_data(SAVE_PATH,a)
            
def load_data(LOAD_PATH):
        with open(LOAD_PATH,"rb") as f:
            load_array=np.load(f)
        return load_array

def save_data(SAVE_PATH,array):
        with open(SAVE_PATH,"wb") as f:
            np.save(f,array)
def normalize_data_sigmoid(SAVE_PATH,LOAD_PATH):
        data=load_data(LOAD_PATH)
        #array_length=data.shape[0]
        array=np.array(data,dtype=np.float64)
        scale = np.array([1.0/180.0, 1.0/255.0, 1.0/255.0],
                       dtype=np.float32) # Scale to fit 0.0 ~ 1.0
        for element in array:
                element[0]*=scale[0]
                element[1]*=scale[1]
                element[2]*=scale[2]
        save_data(SAVE_PATH,array)
def normalize_data(SAVE_PATH,LOAD_PATH):
        means=[]
        stds=[]
        array_length=data.shape[0]
        data=load_data(LOAD_PATH)
        array=np.array(data,dtype=np.float64)
        for index in range(0,array_length):
            im=array[index]
            ch1_mean=np.mean(im[0])
            ch2_mean=np.mean(im[1])
            ch3_mean=np.mean(im[2])
            ch1_std=np.std(im[0])
            ch2_std=np.std(im[1])
            ch3_std=np.std(im[2])
            means.append((ch1_mean,ch2_mean,ch3_mean))
            stds.append((ch1_std,ch2_std,ch3_std))
        mean_running_count=[0,0,0]
        std_running_count=[0,0,0]
        for j in range(array_length):
            for i in range(3):
                mean_running_count[i]+=means[j][i]
                std_running_count[i]+=stds[j][i]
        for element in range(3):
            mean_running_count[element]=mean_running_count[element]/array_length
            std_running_count[element]=std_running_count[element]/array_length
        for element in array:
            for channel in range(3):
                element[channel]-=mean_running_count[channel]
                element[channel]/=std_running_count[channel]
        save_data(SAVE_PATH,array)
        return mean_running_count,std_running_count
def index_generate_random(data_set):
    shuffle_length=data_set.shape[0]
    indexes=np.random.permutation(shuffle_length)
    return indexes
def split_train_test(data,train_size):
    return data[:train_size],data[train_size:]