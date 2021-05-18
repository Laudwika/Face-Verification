# Face_verification
code and how to use the face verification project

## Prerequisites
	-Create a conda environment with python 3.6
	-Install pytorch 1.5.0 with cuda 10.1 (conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch)
	-install scikit-learn = 0.24.2, sckit-image = 0.17.2, openCV-python = 4.5.2.52 through pip
	
# Directories
	-Create a results directory, download and put cur_epoch_test_result.mat inside 
	 https://drive.google.com/drive/folders/1_gn7Nf6OUKy5DtCsN1fZqubKKtggmpPb?usp=sharing
	-Download these models https://drive.google.com/drive/folders/1bV0orFqzf5QCu2hMz9QO51MC1-90SqrA?usp=sharing
	-Create directiories to save and open the images and videos (eg: /data, /video, /result)
	
## Config
Change the information in the config accordingly

	alignment_save = 'test.jpg'                                 #saved aligned face image location
	face_save_loc = 'data/ref/ref.jpg'                          #saved face location, just ignore if using aligned face image
	model18 = 'student-model.h5'                                #resnet18 Model location
	model101 = 'model101.h5'                                    #resnet101 Model location
	mask_model = 'MaskModelBest.h5'                             #resnet18 model location for the mask classifier
	feature_path_loc = './results/cur_epoch_test_result.mat'    #feature location for lfw mean comparison
	face_bank_loc = 'data/features/'                            #feature bank location
	video_output = 'data/zout/clean_test.mp4'                   #video output location, ignore when in production
	model_type = '18'                                           #model type, can use 18 0r 101
	right_image_loc = 'data/a.jpg'                              #Right image location
	right_face_loc = 'data/ac.jpg'                              #Right face location
	left_image_loc = 'data/L,jpg'                               #Left image location
	left_face_loc = 'data/Lc.jpg'                               #Left face location
	vid_config = 0                                              #video type, 0 for live input, otherwise input the video file location
	vid_save_loc = 'data/zout/test.mp4'                         #video save location, ignore when in production
	cur_subject = 's0'                                          #subject for when using presaved feature vectors
	box_constant = 1.5					    #The constant used for validation box size (min 1.5)
	time_to_verify = 2					    #How many detected frames passed to validate (recurring, if 25 then,frame 25, 50, 75, 100, etc)
	
## How to use the classes
initialize the class with the model type ('18' or '101')

	fv = face_verifier(model_type)
	
### Feature extractor
	featureR = fv.image_feature_extractor(right_image_loc, right_face_loc)
send the location image you want to extract feature and the location of where the output image will be

### Image Verification
	featureL = fv.image_feature_extractor()
	featureR = fv.image_feature_extractor(right_image_loc, right_face_loc)
	score = fv.compare_features(featureL, featureR)
	result = fv.verify_result(score, 0.4)
	
	-get left feature and right feature
	-input the 2 features to the compare_features module to get the score
	-input the score and a threshold (set to 0.4 for now) to get result ('SAME' or 'DIFFERENT')

### Video Verification
comment writer and cv2.imshow functions on line 610,611,612 in the face_verification.py

	writer.write(self.img)
	writer2.write(frame)
	cv2.imshow('RetinaFace',self.img)
	
#### if using single reference image
get features

	featureR = fv.image_feature_extractor(right_image_loc, right_face_loc)
	
set vid to 0 for live webcam or set the video location for pre recorded video
send vid and feature into the video_verification module

	fv.video_verification(vid = 0, featureRs = featureR)
	
#### if using from a preprocessed feature bank
get the npy name (subject name, eg: s1, s2, s3, s4)
set vid to 0 for live webcam or set the video location for pre recorded video
send vid and subject name into the video_verification module

	fv.video_verification(vid = 0, subject = cur_subject)

![test-min](https://user-images.githubusercontent.com/70614573/118594658-c32f1000-b7e4-11eb-96db-b9db38d23ab7.gif) 
![test-1](https://user-images.githubusercontent.com/70614573/118595322-d2628d80-b7e5-11eb-9e73-5a3d9725a3f0.gif)






	

	

	

