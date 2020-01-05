# Requirements
#### Built models on tensorflow 1.12, apologies for the older version being used
#### Download models from google drive here - https://drive.google.com/open?id=1GL4RvYcOSBJOKU9H7sjmNI-3XNSYe9Mj - and add this folder to the 'inputs' folder

# Usage:   
### Make sure you have the folder structure as it is in the github repo and you download the models from the above link 
python pcb_errors.py --input_image=1.1.jpg

# Accuracy
My main focus was on True Negative, since the problem statement was to find errors in any pcb, having a high TN is necessary so that we do not miss any defective pcb
