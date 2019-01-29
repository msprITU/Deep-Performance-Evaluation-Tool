### This routin is written by Mustafa Bunyamin Sagman, January 2019 ###
### If you have any problem just send an e-mail to mbunyamins@gmail.com ###

import json
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import re
import os

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']

class_names_in_Coco = {
    '1':'person',
    '2':'bicycle',
    '3':'car',
    '4':'motorcycle',
    '5':'airplane',
    '6':'bus',
    '7':'train',
    '8':'truck',
    '9':'boat',
    '10':'traffic light',
    '11':'fire hydrant',
    '13':'stop sign',
    '14':'parking meter',
    '15':'bench',
    '16':'bird',
    '17':'cat',
    '18':'dog',
    '19':'horse',
    '20':'sheep',
    '21':'cow',
    '22':'elephant',
    '23':'bear',
    '24':'zebra',
    '25':'giraffe',
    '27':'backpack',
    '28':'umbrella',
    '31':'handbag',
    '32':'tie',
    '33':'suitcase',
    '34':'frisbee',
    '35':'skis',
    '36':'snowboard',
    '37':'sports ball',
    '38':'kite',
    '39':'baseball bat',
    '40':'baseball glove',
    '41':'skateboard',
    '42':'surfboard',
    '43':'tennis racket',
    '44':'bottle',
    '46':'wine glass',
    '47':'cup',
    '48':'fork',
    '49':'knife',
    '50':'spoon',
    '51':'bowl',
    '52':'banana',
    '53':'apple',
    '54':'sandwich',
    '55':'orange',
    '56':'broccoli',
    '57':'carrot',
    '58':'hot dog',
    '59':'pizza',
    '60':'donut',
    '61':'cake',
    '62':'chair',
    '63':'couch',
    '64':'potted plant',
    '65':'bed',
    '67':'dining table',
    '70':'toilet',
    '72':'tv',
    '73':'laptop',
    '74':'mouse',
    '75':'remote',
    '76':'keyboard',
    '77':'cell phone',
    '78':'microwave',
    '79':'oven',
    '80':'toaster',
    '81':'sink',
    '82':'refrigerator',
    '84':'book',
    '85':'clock',
    '86':'vase',
    '87':'scissors',
    '88':'teddy bear',
    '89':'hair drier',
    '90':'toothbrush'
}

#### enter the folder where the data is located and the folder where you have the outputs ####
path = "/home/sagman/darknet/coco/"
pathForResult = "/home/sagman/darknet/cocoSonuc/"
target = "train"      #### it can be "val" or "train" because of json name ####

with open (path + 'annotations/instances_' + target + "2014.json") as json_data:
    js = json.load(json_data)
mixed = open(pathForResult + target + 'Mixed.txt','w')
annotations = js['annotations']
number = 0
categoryNumber = 0
for i in annotations:
    liste = [str(0), str(0), str(0), str(0)]
    for j,k in i.items():
        y = []
        if (str(j) == "image_id"):
            liste[0] = str(k)
        elif (str(j) == "category_id"):
            for l,m in class_names_in_Coco.items():
                if (str(l) == str(k)):
                    category = str(m)
                    categoryNumber = class_names.index(str(category))
                    liste[2] = str(categoryNumber)   
        elif (str(j) == "bbox"):
            xmin = int(k[0])
            ymin = int(k[1])
            xmax = xmin + int(k[2])
            ymax = ymin + int(k[3])
            
            y.append(ymin)
            y.append(xmin)
            y.append(ymax)
            y.append(xmax)

            y = str(y)
            y = re.sub(',','', y)
            liste[3] = str(y)
        elif (str(j) == "id"):
            liste[1] = str(k)
        
    str1 = ','.join(str(e) for e in liste)
    mixed.write(str1  + "\n")    
mixed.close()


##### Sorting image_id and id and saving them #####
data = pd.read_csv(pathForResult + target + 'Mixed.txt', sep = ",", header = None)
data.columns = ["image_id", "Ids", "category_id", "bbox"]
with_IDs = data.drop("image_id", axis = 1)
with_Image_Ids = data.drop("Ids", axis = 1)

sort_by_image_id_with_data = data.sort_values("image_id")
sort_by_image_id_with_data.to_csv(pathForResult + target + 'GtWithAll.txt',
 header = None, index = None, sep = ',', mode = 'a')

sort_by_image_id = with_Image_Ids.sort_values("image_id")
sort_by_image_id.to_csv(pathForResult + target + 'GtWithImageId.txt',
 header = None, index = None, sep = ',', mode = 'a')

sort_by_IDs = with_IDs.sort_values("Ids")
sort_by_IDs.to_csv(pathForResult + target + 'GtWithId.txt',
 header = None, index = None, sep = ',', mode = 'a')


##### Preparing the list of names to read in YOLO ####
lines_seen = set()
validation = open(pathForResult + target + 'GtWithImageId.txt','r')
lines = validation.read().split("\n")
del lines[-1]
imageIdForYolo = open(pathForResult + target + 'ImageIdForYolo.txt','w')
imageIdForYolo2 = open(pathForResult + target + 'ImageIdForYolo2.txt','w')
for line in lines:
    datas = line.split(",")
    imageIdForYolo2.write(str(datas[0] + "\n"))
    ids = datas[0].zfill(12)
    if ids not in lines_seen:
        lines_seen.add(ids)
        imageIdForYolo.write(str(ids) + "\n")

validation.close()
imageIdForYolo.close()
imageIdForYolo2.close()


#### Reindexing For Our Rutin ####
lines_seen = set()
infile = open(pathForResult + target + 'ImageIdForYolo2.txt', "r")
outfile = open(pathForResult + target + 'ReindexingForValidation.txt', "w")
string = []
ImageIds = infile.read().split("\n")
del ImageIds[-1]
counter = -1
for i,line in enumerate(ImageIds):
	if line in lines_seen:
		string.append(str(counter))
		lines_seen.add(line)
	elif line not in lines_seen:
		counter = counter + 1
		string.append(str(counter))
		lines_seen.add(line)

##del string[-1]
outfile.write("0" + "\n")
for h in string:
	outfile.write(h + "\n")
outfile.close()

data2 = pd.read_csv(pathForResult + target + 'ReindexingForValidation.txt')
data2.columns = ["new_index"]
data22 = pd.read_csv(pathForResult + target + 'GtWithImageId.txt', sep = ",", header = None)
data22.columns = ["image_id", "category_id", "bbox"]
frames = [data2, data22]
datason = pd.concat(frames, axis = 1)

datason1 = datason.drop("image_id", axis = 1)
datason1.to_csv(pathForResult + target + 'Gt.txt',
 sep = ",", header = None, index = None, mode = 'a')

os.remove(pathForResult + target + 'Mixed.txt')
os.remove(pathForResult + target + 'ReindexingForValidation.txt')
os.remove(pathForResult + target + 'GtWithId.txt')
os.remove(pathForResult + target + 'GtWithAll.txt')
os.remove(pathForResult + target + 'GtWithImageId.txt')
os.remove(pathForResult + target + 'ImageIdForYolo2.txt')