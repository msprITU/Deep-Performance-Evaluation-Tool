### This routin is written by Mustafa Bunyamin Sagman, January 2019 ###
### If you have any problem just send an e-mail to mbunyamins@gmail.com ###

from pydarknet import Detector, Image
import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt

names = open(("/home/sagman/darknet/coco.names"),'r')
lineName = names.read().split("\n")

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))
start_time = time.time()

#path = "/home/mspr/darknet/coco/train2014/"
path = "/home/sagman/darknet/Deneme2/CarDark/img/"
dosya = open(("/home/sagman/darknet/Deneme2/CarDark/results.txt"),'w')
ForRutin = open (("/home/sagman/darknet/Deneme2/CarDark/PredsWithIDs.txt"),'w')

infile = open("/home/sagman/darknet/coco/imageIdForValidation.txt","r")
#infile = open("/home/sagman/darknet/coco/imageIdForTrain.txt","r")
lines = infile.read().split("\n")
#dosya = open(("/home/mspr/darknet/coco" + "/results.txt"),'w')
Files = os.listdir(path)
string1 = path + "results"
if not os.path.exists(string1):
	os.makedirs(string1)
class_id = 0
for i,line in enumerate(lines):
	imagename = "COCO_val2014_" + str(line) + ".jpg"
	#imagename = "COCO_train2014_" + str(line) + ".jpg"
	print(imagename)
	img = cv2.imread(imagename)
	img_darknet = Image(img)
	results = net.detect(img_darknet, thresh = 0.5)

	for cat, score, bounds in results:

		x, y, w, h = bounds
		classes1 = cat.decode("utf-8")
		#classes1 = classes1 + "/n"
		#print(classes1)

		for name in lineName:
			liste = name.split(",")
			if(str(liste[1]) == str(classes1)):
				print (liste[1])
				print(liste[0])
				class_id = int(liste[0])
		xmin = x - w/2
		xmin = round(xmin)
		ymin = y - h/2
		ymin = round(ymin)
		xmax = x + w/2
		xmax = round(xmax)
		ymax = y + h/2
		ymax = round(ymax)

		cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
		cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
		dosya.write(str(line) + "," + str(class_id) + ",[" + str(ymin) + " " + str(xmin) + " " + str(ymax) + " " + str(xmax) + "]" + "\n")
		ForRutin.write(str(i) + "," + str(class_id) + ",[" + str(ymin) + " " + str(xmin) + " " + str(ymax) + " " + str(xmax) + "]" + "\n")
	string = str(string1) + "/" + str(line) + ".jpg"
	#print (string)
	cv2.imwrite(string, img)
			
ForRutin.close()
dosya.close()

