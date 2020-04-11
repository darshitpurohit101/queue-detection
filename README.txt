IMPORTANT LINKS:
1. https://towardsdatascience.com/plug-and-play-object-detection-code-in-5-simple-steps-f1975804373e 
[ here you will find the all explanation about the model and its step for implementation ]
2. https://github.com/matterport/Mask_RCNN
[ you will find code here from above link ]

step1: Annotate the images in the " Image " folder and store exported json file in the same folder
( you can annotate image using http://www.robottos.x.ac.uk/~vgg/software/via/via.html )
step2: Run generating_data.py to get the dataset in required file structure ( give "image" folder path and JSON file path as input ) 
step3: If you want to skip 1 & 2 step then use the Procdata folder from model training.
step4: navigate to Mask_RCNN-master foder > samples > queue
step5: execute queue_detect.py file 
step6: after execution it will ask for following inputs 
 1. enter Train or splash ? (Enter : Train to train the model)
 2. enter path to dataset? (Enter : provide path to dataset that is "procdata" folder containing training and validation data)
 3. enter path for weight file? (Enter : path of weight.h5 file here OR Enter : "last" if you want to continue training from the last iteration)
 4. enter path for log file? (Enter : enter path whereever you want to store the model file after each iteration)


