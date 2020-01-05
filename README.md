# Requirements #
#### Built models on tensorflow 1.12
#### Download models from google drive here - https://drive.google.com/open?id=1GL4RvYcOSBJOKU9H7sjmNI-3XNSYe9Mj - and add this folder to the 'inputs' folder. Will take around 20 mins to download

# Usage 
### Make sure you have the folder structure as it is in the github repo and you download the models from the above link 
python pcb_errors.py --input_image=1.1.jpg

# Accuracy
My main focus was on True Negatives(~>90%), since the problem statement was to find errors in any pcb, having a high TN is necessary so that we do not miss any defective pcb. The disadvantage of focusing on TN was that the False Negatives also started increasing. If there was more time, it should have been possible to decrease FN.

# Approach
Observing all the images under labelImg app gave the idea that all the components places are fixed with very little variations. Additionally, have build the models only for missing parts which are missing in the data provided. Similarly, for rotated parts as well. For eg: compononent 1_a is present through out the data correctly, hence have not built missing or rotated models for this components.
Then segregated the components in four parts:
1. Missing + Rotation (eg: 20_f)
2. Only Missing (eg: 3_b)
3. Only Rotation(eg: 6_6)
4. None (eg: 1_a)

First of all, found the average position of each component on the pcb by averaging it's position across the data provided. Used these average positions to crop the region of interest and test for missing and rotation.

## Missing Approach
Tried a bunch image comparison approaches structural similarity, hashing and mse. But since I was going for an average positioning these approaches gave huge margins for even slight changes in the image. Came back to basics, saw that the pcb will be green when a component is missing, so went ahead and just took a mean of RGB values and compared there difference with a reference image. This reference image has all the components correclty present. 

## Rotation Apprach
For the 16 components which were rotated in atleast one image, following was done for each of th 16 components-
1. VGG16 classification model - Trained only the fc and sigmoid layer. From past experience, this model performs better with very few data so went for this straight away. Trained the data with Adam optimizer for a lr=0.00001. Used Keras Image_Data_Generator for augmentation
2. Overfitted the data on this model (Since we wanted high TN)
3. Training samples
  a. Train: correct: ~20 rotated:(as many there for a component, mostly 1/2)
  b. Test: correct: 4 rotated:same as Train
