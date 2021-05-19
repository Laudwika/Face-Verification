import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision
import os

from torchvision import transforms
from networks.models import resnet18, resnet101, create_retinaface
import torchvision.transforms as transforms
import time
import scipy.io
import sklearn.metrics
from helper.align_faces import warp_and_crop_face, get_reference_facial_points
import helper.config as config
import helper.utils as utils
import helper.image_helper as image_helper

#-*- coding: utf-8 -*-

class face_verifier():
    #Initialize variables
    def __init__(self, model_type = '18'):
        ## SET DEVICE ##
        self.device = utils.device

        ## RETINA FACE / FACE DETECTOR ##

        # Create torchvision model
        return_layers = {'layer2':1,'layer3':2,'layer4':3}
        self.RetinaFace = create_retinaface(return_layers)

        # Load trained model
        retina_dict = self.RetinaFace.state_dict()
        pre_state_dict = torch.load(config.retinafacemodel)
        pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
        self.RetinaFace.load_state_dict(pretrained_dict)

        self.RetinaFace = self.RetinaFace.to(self.device)
        self.RetinaFace.eval()

        ## ARCFACE / FACE VERIFICATION ##

        # Load model

        if model_type == '18':
            self.model = resnet18(use_se = True)

            # Freeze Model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.to(self.device)

            # Load pretrained model
            self.model.load_state_dict(torch.load(config.model18))
            
        else:
            self.model = resnet101(use_se = True)

            # Freeze Model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.to(self.device)

            # Load pretrained model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(config.model101))

        self.model = self.model.eval()

        ## MASK CLASSIFIER ##

        #Load Model
        self.model_conv = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.model_conv.fc.in_features
        #Create new final layer
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc_1 = nn.Linear(num_ftrs, 2)
            def forward(self, x):
                
                x = self.fc_1(x)
                return x

        net = Net()
        self.model_conv.fc = net
        self.model_conv.bn3 = nn.BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #Load pretrained model
        self.model_conv.load_state_dict(torch.load(config.mask_model))
        self.model_conv = self.model_conv.to(self.device)
        self.model_conv = self.model_conv.eval()

        ## CALCULATE MU FROM LFW ##
        # feature_path='./results/cur_epoch_result.mat'
        feature_path=config.feature_path_loc
        result = scipy.io.loadmat(feature_path)        
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']
        valFold = fold != 11
        flags = np.squeeze(flags)

        # Calcluate mu
        self.mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        self.mu = np.expand_dims(self.mu, 0)
        
        ## SET VARIABLES ##
        self.threshold = config.con_threshold           #Thresholding value, set to 0.4
        self.fail = 0                                   #For analysis purposes (total frames with verification failures)
        self.lowest = 0.0                               #For analysis purposes (lowest score)
        self.highest = 0.0                              #For analysis purposes (highest score)
        self.mean = 0.0                                 #For analysis purposes (average score)
        self.scores = []                                #For analysis purposes (score array)
        self.w = None                                   #Video width size
        self.h = None                                   #video height size
        self.validate_box = True                        #Validation box phase
        self.num_box_fail = 0                           #Timer to restart the temporal consistency
        self.valid_box = [0, 0, self.w, self.h]         #Validation for bounding box
        self.validation_bounding_box = [0, 0, 0, 0]     #Bounding box for temporal consistincy
        self.start = 1                                  #Value to see how many frames has passed
        self.valface = 0                                #Value to see how many frames with detected faces has passed
        self.score = 0.0                                #Verification similarity score
        self.img = None                                 #Image to send through functions
        self.flag = 2                                   #Flag (is same or not)
        self.change_feature = config.change_features    #Change feature when value is a certain similatiry
        self.featureLs = None                           #"Left" feature
        self.featureRs = None                           #"Right" feature
        self.time_taken = []                            #For analysis purposes (time taken to verify)
        self.is_anchor = False                          #Get anchor or return anchor phase
   
    #Load images into cv2 and resize
    def img_loader(self, path):
        try:
            with open(path, 'rb') as f:
                img = cv2.imread(path)
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = cv2.resize(img, (112,112), interpolation = cv2.INTER_AREA)
                return img
        except IOError:
            print('Cannot load image ' + path)

    #Verify function returns the features used in feature extractor
    def feature_extractor(self, face):
        #set variables
        featureLs = None
        model = self.model.eval()
        im1 = face
    
        ## INITIALIZE THE IMAGES ##
        image_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = self.img_loader(im1)
        data = [img, cv2.flip(img,1)]

        # Transform the data
        for i, img in enumerate(data):

            data[i] = image_transforms(data[i])
            data[i] = data[i].unsqueeze(0)
            data[i] = data[i].to(self.device)

        start_time = time.time() #To validate time

        #Get features
        with torch.no_grad():
            res = [model(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        end_time = time.time() #To validate time

        # self.time_taken.append((end_time - start_time)) #To validate time

        return featureL

    #Feature comparison
    def compare_features(self, featureLs = [], featureRs = []):

        if featureLs == []:
            featureLs = self.featureLs
            featureRs = self.featureRs
        

        string1 = 'LFW distance         : '
        string2 = 'Euclidiean           : '
        string3 = 'Cosine Distance      : '
        string4 = 'Combination Distance : '
        #Compare input feature with ref feature
        scores = utils.evaluation(featureLs, featureRs, self.mu)
        scores = abs(scores[0])
        # print(string1, scores)

        featureLcos = featureLs.reshape(1,-1)
        featureRcos = featureRs.reshape(1,-1)

        #Euclidean
        # diff = np.subtract(self.featureLs, self.featureRs)
        # dist = np.sum(np.square(diff),1)
        # print(string2, dist[0])

        diff = np.subtract(featureLcos, featureRcos)
        dist = np.sum(np.square(diff),1)
        dist[0] = 1 - (dist[0] * 5)


        cos = sklearn.metrics.pairwise.cosine_similarity(featureLcos, featureRcos, dense_output=False)
        # print(string3, cos[0][0])
        com_scores = (cos[0][0] + scores) / 2


        #Verification Validation (optional), if there is 1 above a certain threshold, that becomes the new reference feature
        if scores > 0.999 and self.change_feature == False:
            self.featureRs = self.featureLs
            change_feature = True
        

        return com_scores

    #Draw on video
    def draw(self, box, flag, label):
        img = self.img
        
        score = self.score
        flag = flag
        validation_bounding_box = self.validation_bounding_box
        # validation_bounding_box = self.validation_bounding_box
        ## SHOW LANDMARKS/VISUALIZATION (OPTIONAL) ##
        # cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
        # cv2.circle(img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
        # cv2.circle(img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
        # cv2.circle(img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
        # cv2.circle(img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
        try:
            if label == 'Mask':
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,192,203),thickness=2)
                cv2.rectangle(img,(validation_bounding_box[0],validation_bounding_box[1]),(validation_bounding_box[2],validation_bounding_box[3]),(255,0,0),thickness=2)
                cv2.putText(img=img, text='{}'.format(label), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                ## FLAG 0 = SAME      ##
                ## FLAG 1 = DIFFERENT ##
                ## FLAG 2 = WAITING   ##

                if flag == 0:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)
                    cv2.rectangle(img,(validation_bounding_box[0],validation_bounding_box[1]),(validation_bounding_box[2],validation_bounding_box[3]),(255,0,0),thickness=2)

                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                elif flag == 1:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),thickness=2)
                    cv2.rectangle(img,(validation_bounding_box[0],validation_bounding_box[1]),(validation_bounding_box[2],validation_bounding_box[3]),(255,0,0),thickness=2)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                elif flag == 2:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),thickness=2)
                    cv2.rectangle(img,(validation_bounding_box[0],validation_bounding_box[1]),(validation_bounding_box[2],validation_bounding_box[3]),(255,0,0),thickness=2)
                    cv2.putText(img=img, text='{}'.format(label), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        except:
            pass
        return img

    #Mask detect
    def mask_detection(self, face):
        model_conv = self.model_conv
        
        ## PROCESS THE IMAGE ##
        image_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # image_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        coler_coverted = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(coler_coverted)
        img = image_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        ## GET PREDICTION RESULT ##
        prediction = model_conv(img)
        _, preds = torch.max(prediction,1)
        predictions = preds.cpu().numpy()[0]
        
        return predictions

    #Face detection for image input, not video frames, no temporal consistency, assumed single person
    def image_face_detector(self, picked_boxes, picked_landmarks, picked_scores):
        #return the first box detected
        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    return True, landmark
            else:
                return False, None

    #Face detector Function(RetinaFace)
    def face_detector(self):
        self.img = torch.from_numpy(self.img)
        self.img = self.img.permute(2,0,1)

        input_img = self.img.unsqueeze(0).float().to(self.device)
        picked_boxes, picked_landmarks, picked_scores = utils.get_detections(input_img, self.RetinaFace, score_threshold=0.5, iou_threshold=0.3)

        np_img = self.img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        self.img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)
        
        return picked_boxes, picked_landmarks, picked_scores

    #Get anchor for validation box
    def compute_anchor(self, picked_boxes, picked_landmarks, picked_scores):
        #initialize values
        face_save = config.alignment_save

        max_score = -999999999999999.9
        max_index = 0
        is_face = False
        self.validation_bounding_box = 0, 0, self.w, self.h
        #Iterate through the picked boxes
        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    face = self.img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    
                    cv2.imwrite(face_save, face)
                    prediction = 1
                    #Detect mask first, if masked then skip anchoring for now
                    prediction = self.mask_detection(face)
                
                    label = "Mask" if prediction == 0 else "No Mask"

                    if label == 'Mask':
                        
                        return False, None, None, None, None, None, None, None, None, None
                    else:

                        #If not then start the verification process for every detected face
                        face = face_save
                        start_time = time.time()
                        self.featureLs = self.feature_extractor(face)
                        score = self.compare_features()
                        
                    
                        #If sclassified as the same person then set as anchor
                        
                        if score > max_score:
                            max_score = score
                            self.score = max_score
                            max_index = j
                            box = [int(x) for x in box]
                            is_face = True
                            if score > self.threshold:

                                self.is_anchor = True
        
                validation_bounding_box = image_helper.resize_box(self.w, self.h, box)
                return is_face, box, validation_bounding_box, landmark
            else:
                return False, None, None, None
                                               
    #Return anchored boxes
    def return_anchor(self, picked_boxes, picked_landmarks, picked_scores):
        #Same process as the compute anchor, but within the previously picked box
        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    #New detected box has to be valid (inside the validation bounding box)
                    #If video just start then set to default size, else use previous valid
                    if self.valface == 0:
                        self.validation_bounding_box = [0, 0, self.w, self.h ]
                    if self.valface > 0:
                        self.validation_bounding_box = image_helper.resize_box(self.w, self.h, box)
                        
                    #verify if the box is valid or not
                    validate_box = image_helper.verify_box(box, self.valid_box)

                    #If box is valid then set new validation boxes and continue

                    if validate_box == True:
                        self.valid_box = self.validation_bounding_box
                        self.num_box_fail = 0
                        return True, box, self.validation_bounding_box, landmark

                    #If invalid then continue trying to search, but don't verify
                    if validate_box == False:
                        self.num_box_fail += 1
                        #if failed for a certain amount of times then restart the anchoring process
                        if self.num_box_fail == config.num_box_fail:
                            self.num_box_fail = 0
                            self.valid_box = [0, 0, self.w, self.h]
                            self.valface = 0
                            validate_box = True
                    
        return False, None, None, None

    #Video verification function (input video, uses feature as reference)
    def video_verification(self, vid = config.vid_config, save_vid = config.vid_save_loc , featureRs = [], subject = None):
        #Initialize values
        if subject:
            self.featureRs = utils.get_feature_bank(subject = subject)
        else:
            self.featureRs = featureRs
        save_vid = save_vid
        face_save = config.face_save_loc
        cap = cv2.VideoCapture(vid)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #########################
        ##FOR ANALYSIS PURPOSES##
        writer = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (self.w,self.h))
        writer2 = cv2.VideoWriter(config.video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (self.w,self.h))
        #########################

        self.valid_box = [0, 0, self.w, self.h]
        self.is_anchor = False
        facial5points = [0,0,0,0,0,0,0,0,0,0]
        landmarks = [0,0,0,0,0,0,0,0,0,0]
        box = [0, 0, 0, 0]
        
        i = 0
        label = 'No Mask'
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:

                start_time = time.time()

                is_face = False

                self.img = frame
                self.img = np.copy(frame[:, :, ::-1])
                #Detect faces from the frame
                picked_boxes, picked_landmarks, picked_scores = self.face_detector()
                #Start anchoring process or continue with temporaly consistent tracking
                if self.is_anchor == False:
                    is_face, box, self.validation_bounding_box, landmark = self.compute_anchor(picked_boxes, picked_landmarks, picked_scores)
 
                elif self.is_anchor == True:
                    is_face, box, self.validation_bounding_box, landmark = self.return_anchor(picked_boxes, picked_landmarks, picked_scores)

                #only continue if face is detected
                if is_face == True and landmarks is not None:
                    box = [int(x) for x in box]
                    #Start the alignment process
                    landmark = landmark.detach().cpu()

                    for l in range(10):
                        landmarks[l] = int(landmark[l].item())
                    output_size = (112,112)
                    #Alignment algorithm
                    facial5points = [landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8], landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]]

                    facial5points = np.reshape(facial5points, (2, 5))

                    default_square = True
                    inner_padding_factor = 0.25
                    outer_padding = (0, 0)

                    #Get the reference 5 landmarks position 
                    reference_5pts = get_reference_facial_points(
                        output_size, inner_padding_factor, outer_padding, default_square)
                    #Save the alignmed image
                    
                    dst_img = warp_and_crop_face(self.img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
                    cv2.imwrite(config.alignment_save, dst_img)
                    
                    ##################################################
                    ##FOR ANALYSIS PURPOSES, TO SHOW THE BOUNDING BOX
                    face = frame[box[1]:box[3], box[0]:box[2]]
                    cv2.imwrite(face_save, face)
                    ##################################################

                    #Set mask is true first, if undetected then it will pass
                    prediction = 1
                    #Detect the mask
                    prediction = self.mask_detection(face)
                
                    label = "Mask" if prediction == 0 else "No Mask"

                    if label == 'Mask':
                        pass
                    else:
                        #if unmasked the start to verify after a certain amount of detected frames
                        if self.valface > config.time_to_verify:
                            face = config.alignment_save
                            start_time = time.time()
                            self.featureLs = self.feature_extractor(face)
                            self.score = self.compare_features()
                            
                            result = utils.verify_result(self.score, self.threshold)
                            self.valface = 1

                            if result == 'SAME':
                                self.flag = 1
                            else:
                                self.flag = 0
                                self.is_anchor = False
                                self.valid_box = [0, 0, self.w, self.h]



                #How many frames passed and how many frames with detected faces passed
                self.start += 1
                self.valface += 1

                #############################################
                ##FOR ANALYSIS PURPOSES, INFERENCE TIME TAKE
                end_time = time.time()
                self.time_taken.append((end_time - start_time)) #To validate time
                #############################################

                ###########################################
                ##FOR ANALYSIS PURPOSES, VISUALIZATION
                self.img = self.draw(box, self.flag, label)
                writer.write(self.img)
                writer2.write(frame)
                cv2.imshow('RetinaFace',self.img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                i += 1
                ###########################################
            
                # except:
                #     print('passed')
                #     pass
            else:
                break
        
        ##################################################################
        ##FOR ANALYSIS PURPOSES, VISUALIZATION AND TIME INFERENCE ANALYSIS
        cap.release()
        writer.release()
        writer2.release()
        cv2.destroyAllWindows()



        self.time_taken.pop(0)

        
        print()
        ave = sum(self.time_taken) / len(self.time_taken)
        print(ave)
        print(min(self.time_taken))
        print(max(self.time_taken))    
        ##################################################################


    #Return features from a single image
    def image_feature_extractor(self, image_loc = config.left_image_loc, face_loc = config.left_face_loc):
        #Initialize values
        box = [0,0,0,0]
        image_loc = image_loc
        self.img = cv2.imread(image_loc)

        self.img = np.copy(self.img[:, :, ::-1])
        self.h, self.w, c = self.img.shape
        self.valid_box = [0, 0, self.w, self.h]
        
        #Detect the face
        picked_boxes, picked_landmarks, picked_scores = self.face_detector()
        is_face, landmarks  = self.image_face_detector(picked_boxes, picked_landmarks, picked_scores)
        output_size = (112,112)
        #Align the faces
        facial5points = [landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8], landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]]
        if is_face == False:
            return None
     
        else:
            facial5points = np.reshape(facial5points, (2, 5))

            default_square = True
            inner_padding_factor = 0.25
            outer_padding = (0, 0)

            # get the reference 5 landmarks position in the crop settings
            reference_5pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding, default_square)

            # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
            dst_img = warp_and_crop_face(self.img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
            cv2.imwrite(face_loc, dst_img)
            #Input into feature extractor function to get the feature vector
            return self.feature_extractor(face_loc)


if __name__=='__main__':


    # fv = face_verifier(model_type = '18')
    fv = face_verifier(config.model_type)

    # summary(fv.model, (3,112,112))
    # exit()

    #Vid = 0 for live webcam input, else put in video location
    vid = 0
    # featureL = fv.image_feature_extractor()
    featureR = fv.image_feature_extractor(config.right_image_loc, config.right_face_loc)
    
    # fv.image_feature_extractor(image_loc='data/a.jpg')

    fv.video_verification(vid = 0, featureRs = featureR)
    # fv.video_verification(subject = cur_subject)
    # fv.image_verification()

    exit()
    featureL = fv.image_feature_extractor()
    featureR = fv.image_feature_extractor(config.right_image_loc, config.right_face_loc)

    score = fv.compare_features(featureL, featureR)
    result = utils.verify_result(score, 0.4)
    print(score, result)

    
