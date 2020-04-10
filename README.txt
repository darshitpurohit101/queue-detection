step1: Run generating_data.py to get the dataset in required file structure
step2: navigate to Mask_RCNN-master foder > samples > queue
step3: execute queue_detect.py file 
step4: after execution it will ask for following inputs 
 1. enter Train or splash ? (Enter : Train to train the model)
 2. enter path to dataset? (Enter : provide path to dataset that is folder containing training and validation data)P
 3. enter path for weight file? (Enter : path of weight.h5 file here OR Enter : "last" if you want to continue training from the last iteration)
 4. enter path for log file? (Enter : enter path whereever you want to store the model file after each iteration)

IMPORTANT LINKS:
1. https://towardsdatascience.com/plug-and-play-object-detection-code-in-5-simple-steps-f1975804373e 
[ here you will find the all explanation about the model and its step for implementation ]
