## Instructions to use the evaluation routine on other object detection models:

This project aims to provide the results in COCO dataset for different object detection models styles like Masked R-CNN, YOLO  & SSD.

<!--- There will be the explanation and the abstract for the project here. --->

## Using Mask-RCNN

When Mask-RCNN makes a prediction, how reliable are the results? When the model is trained, does it perform the same for all classes it is trained for? How do the number of instances in the training set per class affect the overall prediction accuracy? Is more better? These are some of the questions that can not be easily answered just by merely looking at the mIoU score of the network. In this analysis, Mask-RCNN model, trained with COCO dataset with 80k images, is evaluated on official train and test splits of the COCO dataset to attempt to answer these questions.

A notebook with the demo for Mask-RCNN can be found in [demo/Evaluate_Models.ipynb](demo/Evaluate_Models.ipynb).

## Object Detection by YOLO

In this project we aimed to report the performance of YOLO on COCO dataset. In COCO dataset there are approximately 81k training images and 41k validation images. Detailed descriptions can be found from the [COCO dataset's homepage](http://cocodataset.org/#home). 

YOLOv3-608 is used as the model to see how YOLO provides performance in training and validation images. I chose this model because if we look at other models of YOLO, we can observe that YOLOv3-608 is better than other models according to [homepage of YOLO](https://pjreddie.com/darknet/yolo/). I used darknet, which is the backbone of YOLO with python interpreter. Also I used OpenCV library with python interpreter. 

The steps of making the project can be divided into 3, which are:
* Getting ground truth (GT) information from COCO dataset and obtaining image names.
* Extraction of predictions which are made by YOLO for training and validation images. 
* Measuring performance of YOLO with reporting YOLO's predictions according to ground truth data.

### Installing Neccessary Environments

For using codes in this file, you must install neccessary libraries. First of all download the [requirements.txt](YOLO/Setup/requirements.txt) file. After that; for installiation of the libraries, you can use the command in terminal down below.

```
pip3 install -r requirements.txt
```
and all environments will be installed automatically.

We  use ipython notebook to look for performance of YOLO. To do this first you need to install Jupyter Notebook from terminal as follows
```
pip3 install jupyter
```
and it will be installed automatically.


### Adjusting the Ground Truth and Image Id Files

For arranging this files, I write [CocoAnnotations.py](YOLO/Setup/CocoAnnotations.py) script to generate necessary files. (The file can be downloaded with the link given. ) However, before using this code, you must set annotation files path and you must enter the annotation file name (train or val) by changing target variable.
You have to copy the training and validation info in json format to annotion directory. Currently we use COCO instances_train2014.json and instances_val2014.json file which can be downloaded from [here.](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

After adjusting python file you can run this script from terminal by writing

```
python3 CocoAnnotations.py
```

Python code will generate the following files. I've added descriptions of the files.

* [trainGT.txt](YOLO/Files/trainGt.txt) : Inside this file there are training image Id's, corresponding bounding boxes with class level information.

* [valGT.txt](YOLO/Files/valGt.txt) : Inside this file there are validaiton image Id's, corresponding bounding boxes with class level information.

* [trainImageIdForYolo.txt](YOLO/Files/trainImageIdForYolo.txt) : This file contains the name list of the COCO training dataset image.

* [valImageIdForYolo.txt](YOLO/Files/valImageIdForYolo.txt) : This file contains the name list of the COCO validation dataset image.

The links to these text files are set as an example in the YOLO directory.

### Running YOLO for Getting Predictions

You need to run [YoloPrediction.py](YOLO/Setup/YoloPrediction.py) to generate the prediction files (use image id txt files (trainImageIdForYolo.txt or valImageIdForYolo.txt) which are created in the previous step, don't forget to change paths to access to YOLO config file and YOLO weight file). After setting these variables you can run the code from terminal by writing

```
python3 YoloPrediction.py
```

This code generate the following files. I've added descriptions of the files.

* [predstrain.txt](YOLO/Files/predstrain.txt) : This file contains the predictions made by YOLO for training images.

* [predsval.txt](YOLO/Files/predsVal.txt) : This file contains the predictions made by YOLO for validation images.

The links are also given here, to set an example; can be found in the YOLO directory.

### Running Evaluation Tool For Obtaining YOLO Performance

Before running evaluation tool, we need to make following arrangements.

* Copy trainGT.txt file to trainlogs directory and change its name to gt.txt

* Copy predstrain.txt file to trainlogs directory and change its name to preds.txt

* Copy valGT.txt file to vallogs directory and change its name to gt.txt

* Copy predsval.txt file to vallogs directory and change its name to preds.txt

Switch to directory of [Evaluate_Models.ipynb](YOLO/Evaluation/Evaluate_Models.ipynb) file and write terminal to jupyter lab. Run Evaluate_Models.ipynb to generate performance results.



<!--- You only look once (YOLO) is a system for detecting objects on the Pascal VOC 2012 dataset. It can detect the 20 Pascal object classes: --->

<!--- person
bird, cat, cow, dog, horse, sheep
aeroplane, bicycle, boat, bus, car, motorbike, train
bottle, chair, dining table, potted plant, sofa, tv/monitor
YOLO is joint work with Santosh, Ross, and Ali, and is described in detail in our paper. --->

<!---How it works
All prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections. --->

<!---We use a totally different approach. We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. --->

<!---This also applies for YOLO system. *** --->

<!---A notebook that includes YOLO results can be found at [demo/Evaluate_Models2.ipynb](demo/Evaluate_Models.ipynb). --->

<!--- A notebook with the demo for YOLO can be found in [demo/Evaluate_Models2.ipynb](demo/Evaluate_Models.ipynb). --->

## Object Detection by SSD

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

