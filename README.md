## Instructions to use the evaluation routine on other object detection models:

This project aims to provide the results in COCO dataset for different object detection models styles like Masked R-CNN, YOLO  & SSD.

There will be the explanation and the abstract for the project here.

## Using Mask-RCNN

When Mask-RCNN makes a prediction, how reliable are the results? When the model is trained, does it perform the same for all classes it is trained for? How do the number of instances in the training set per class affect the overall prediction accuracy? Is more better? These are some of the questions that can not be easily answered just by merely looking at the mIoU score of the network. In this analysis, Mask-RCNN model, trained with COCO dataset with 80k images, is evaluated on official train and test splits of the COCO dataset to attempt to answer these questions.

A notebook with the demo for Mask-RCNN can be found in [demo/Evaluate_Models.ipynb](demo/Evaluate_Models.ipynb).

## Using YOLO Object Detection System

You only look once (YOLO) is a system for detecting objects on the Pascal VOC 2012 dataset. It can detect the 20 Pascal object classes:

person
bird, cat, cow, dog, horse, sheep
aeroplane, bicycle, boat, bus, car, motorbike, train
bottle, chair, dining table, potted plant, sofa, tv/monitor
YOLO is joint work with Santosh, Ross, and Ali, and is described in detail in our paper.

How it works
All prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

This also applies for YOLO system. *** 

A notebook with the demo for YOLO can be found in [demo/Evaluate_Models2.ipynb](demo/Evaluate_Models.ipynb).

## Using SSD Object Detection System

This also applies for SSD object detection model style. *** 

A notebook with the demo for SSD can be found in [demo/Evaluate_Models3.ipynb](demo/Evaluate_Models.ipynb).




To work with .ipynb notebook all you need to know. ...


You can easily evaluate your different object detection models on any dataset with Evaluate_Models notebook. To do this, predictions of your model and groundtruth information must be saved separately in .txt files and match the below format:



**image_id,class_id,[y1 x1 y2 x2]* **

*: bounding box coordinates are not in a python list but numpy array. If you have a list, simply convert it to numpy array for convenience.


Each line which stores a single bounding box information, must be ordered by the image_id. Examples of the expected input files are provided under the *demo/vallogs* directory with the names of **"gt.txt"** and **"preds.txt"**. Since it takes some time to obtain the output of the model for every single image in big datasets like COCO, it is crucial to save the results and  is the reason why this routine needs them as input files.

After generating the input files, simply open *Evaluate_Models.ipynb* and follow the instructions in the notebook to evaluate your own models. You may change the parameters such as IoU_threshold according to your needs as explained in the notebook. Plottings and figures will be created and saved under this directory.


### Getting a Result:

You can directly access the output video of our results [here](output.avi)

