import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import torchvision_model
import os
from os import listdir
from os.path import isfile, join, isdir
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from models import resnet18, resnet101
import random
import torchvision.transforms as transforms
import time
import math
import scipy.io
from pathlib import Path
from math import ceil as r 
import sklearn.metrics
from sklearn.metrics.pairwise import cosine_similarity
from align_faces import warp_and_crop_face, get_reference_facial_points
from config import alignment_save, face_save_loc, model18, model101, mask_model, feature_path_loc, face_bank_loc, video_output
from config import model_type, right_image_loc, right_face_loc, left_image_loc, left_face_loc, vid_config, vid_save_loc, cur_subject, time_to_verify
from utils import device, verify_box, resize_box, get_detections

#-*- coding: utf-8 -*-

class face_verifier():
    #Initialize variables
    def __init__(self, model_type = '18'):
        ## SET DEVICE ##
        self.device = device

        ## RETINA FACE / FACE DETECTOR ##

        # Create torchvision model
        return_layers = {'layer2':1,'layer3':2,'layer4':3}
        self.RetinaFace = torchvision_model.create_retinaface(return_layers)

        # Load trained model
        retina_dict = self.RetinaFace.state_dict()
        pre_state_dict = torch.load('model.pt')
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
            self.model.load_state_dict(torch.load(model18))
            
        else:
            self.model = resnet101(use_se = True)

            # Freeze Model
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.to(self.device)

            # Load pretrained model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(model101))

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
        self.model_conv.load_state_dict(torch.load(mask_model))
        self.model_conv = self.model_conv.to(self.device)
        self.model_conv = self.model_conv.eval()

        ## CALCULATE MU FROM LFW ##
        # feature_path='./results/cur_epoch_result.mat'
        feature_path=feature_path_loc
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
        self.threshold = 0.4
        self.vid = None
        self.fail = 0
        self.lowest = 0.0
        self.highest = 0.0
        self.mean = 0.0
        self.scores = []
        self.zoomvid = self.vid
        self.w = None
        self.h = None
        self.validate_box = True
        self.num_box_fail = 0
        self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h            
        self.start = 1
        self.valface = 0
        self.score = 0.0
        self.img = None
        self.flag = 2
        self.change_feature = False
        self.featureLs = None
        self.featureRs = None
        self.time_taken = []
        self.out = []
        self.is_anchor = False

    #Compute scores
    def evaluation(self, featureL, featureR, threshold):
        featureL = featureL
        featureR = featureR

        ## COMPARE FEATURES ##
        featureL = featureL - self.mu
        featureR = featureR - self.mu
        featureL = featureL / np.expand_dims(np.sqrt(np.sum(np.power(featureL, 2), 1)), 1)
        featureR = featureR / np.expand_dims(np.sqrt(np.sum(np.power(featureR, 2), 1)), 1)

        ## GET SCORES ##
        scores = np.sum(np.multiply(featureL, featureR), 1)

        return scores
            
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

    #Verify function
    def verify(self, face):
        start = self.start
        featureLs = None
        featureRs = self.featureRs
        threshold = self.threshold
        model = self.model.eval()
        device = self.device
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
        scores = self.evaluation(featureLs, featureRs, self.threshold)
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

    #Verify results same or different
    def verify_result(self, scores, threshold):
        if scores > threshold:
            result = 'SAME'
        else:
            result = 'DIFFERENT'
        # print(result)
        return result

    #Draw on video
    def draw(self, box, flag, box0, box1, box2, box3, label):
        img = self.img
        
        score = self.score
        box = box
        flag = flag

        ## SHOW LANDMARKS/VISUALIZATION (OPTIONAL) ##
        # cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
        # cv2.circle(img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
        # cv2.circle(img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
        # cv2.circle(img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
        # cv2.circle(img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
        try:
            if label == 'Mask':
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,192,203),thickness=2)
                cv2.rectangle(img,(box0,box1),(box2,box3),(255,0,0),thickness=2)
                cv2.putText(img=img, text='{}'.format(label), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                ## FLAG 0 = SAME      ##
                ## FLAG 1 = DIFFERENT ##
                ## FLAG 2 = WAITING   ##

                if flag == 0:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255),thickness=2)
                    cv2.rectangle(img,(box0,box1),(box2,box3),(255,0,0),thickness=2)

                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                elif flag == 1:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),thickness=2)
                    cv2.rectangle(img,(box0,box1),(box2,box3),(255,0,0),thickness=2)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(img=img, text='similiarity: {:.2f}%'.format(score*100), org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                elif flag == 2:
                    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),thickness=2)
                    cv2.rectangle(img,(box0,box1),(box2,box3),(255,0,0),thickness=2)
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

    #Image face detector
    def image_face_detector(self):
        self.img = torch.from_numpy(self.img)
        self.img = self.img.permute(2,0,1)

        input_img = self.img.unsqueeze(0).float().to(self.device)
        picked_boxes, picked_landmarks, picked_scores = get_detections(input_img, self.RetinaFace, score_threshold=0.5, iou_threshold=0.3)

        np_img = self.img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        self.img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(self.img)

        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    if self.valface == 0:
                        box0, box1, box2, box3 = 0, 0, self.w, self.h 
                    if self.valface > 0:
                        box0, box1, box2, box3 = resize_box(self.w, self.h, box[0], box[1], box[2], box[3])
                    
                    validate_box = verify_box(box[0], self.vbox0, box[1], self.vbox1, box[2], self.vbox2, box[3], self.vbox3)


                    if validate_box == True:
                        self.vbox0, self.vbox1, self.vbox2, self.vbox3 = box0, box1, box2, box3
                        self.num_box_fail = 0
                        return True, int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box0), int(box1), int(box2), int(box3), landmark


                    if validate_box == False:
                        self.num_box_fail += 1
                        if self.num_box_fail == 100:
                            self.num_box_fail = 0
                            self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h
                            self.valface = 0
                            validate_box = True
                    
        return False, None, None, None, None, None, None, None, None, None

    #Face detector Function(RetinaFace)
    def face_detector(self):
        self.img = torch.from_numpy(self.img)
        self.img = self.img.permute(2,0,1)

        input_img = self.img.unsqueeze(0).float().to(self.device)
        picked_boxes, picked_landmarks, picked_scores = get_detections(input_img, self.RetinaFace, score_threshold=0.5, iou_threshold=0.3)

        np_img = self.img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        self.img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)
        
        return picked_boxes, picked_landmarks, picked_scores

    #Get anchor for validation box
    def compute_anchor(self, picked_boxes, picked_landmarks, picked_scores):
        # print('inside_anchor')
        face_save = alignment_save

        max_score = -999999999999999.9
        max_index = 0
        is_face = False
        bbox0, bbox1, bbox2, bbox3 = 0, 0, self.w, self.h
        # print('inside')
        # print(len(picked_boxes))
        for j, boxes in enumerate(picked_boxes):
            # print(j)
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    face = self.img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    
                    cv2.imwrite(face_save, face)
                    prediction = 1

                    prediction = self.mask_detection(face)
                
                    label = "Mask" if prediction == 0 else "No Mask"

                    if label == 'Mask':
                        
                        return False, None, None, None, None, None, None, None, None, None
                    else:
                        face = face_save
                        start_time = time.time()
                        self.featureLs = self.verify(face)
                        score = self.compare_features()
                        result = self.verify_result(self.score, self.threshold)
                    
                    
                        if score > self.threshold:
                            # print(self.is_anchor)

                            self.is_anchor = True
                        
                        if score > max_score:
                            # print(self.is_anchor)
                            max_score = score
                            self.score = max_score
                            max_index = j
                            box0, box1, box2, box3 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            is_face = True
                            # print(score)
                # print('max:', max_index)
                # print(box0, box1, box2, box3)
                bbox0, bbox1, bbox2, bbox3 = resize_box(self.w, self.h, box0, box1, box2, box3)
                return is_face, box0, box1, box2, box3, bbox0, bbox1, bbox2, bbox3, landmark
            else:
                return False, None, None, None, None, None, None, None, None, None
                
                            
            # result = self.verify_result(self.score, self.threshold)
                    
    #Return anchored boxes
    def return_anchor(self, picked_boxes, picked_landmarks, picked_scores):
        # print('inside')
        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box, landmark, conf in zip(picked_boxes[j],picked_landmarks[j],picked_scores[j]):
                    if self.valface == 0:
                        box0, box1, box2, box3 = 0, 0, self.w, self.h 
                    if self.valface > 0:
                        box0, box1, box2, box3 = resize_box(self.w, self.h, box[0], box[1], box[2], box[3])
                    
                    validate_box = verify_box(box[0], self.vbox0, box[1], self.vbox1, box[2], self.vbox2, box[3], self.vbox3)

                    # print(validate_box)

                    if validate_box == True:
                        self.vbox0, self.vbox1, self.vbox2, self.vbox3 = box0, box1, box2, box3
                        self.num_box_fail = 0
                        return True, int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box0), int(box1), int(box2), int(box3), landmark


                    if validate_box == False:
                        self.num_box_fail += 1
                        if self.num_box_fail == 25:
                            self.num_box_fail = 0
                            self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h
                            self.valface = 0
                            validate_box = True
                    
        return False, None, None, None, None, None, None, None, None, None

    #Get features from a feature bank
    def get_feature_bank(self, subject = 's36'):
        ## FEATURE LOADER / NPY ##
        subject_path = face_bank_loc + subject + '.npy'
        filename = Path(subject_path)

        # Open feature file
        with filename.open('rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            self.out = np.load(f)
        
            while f.tell() < fsz:
                self.out = np.vstack((self.out, np.load(f)))

        # Choose random feature base
        rf = random.randint(0, len(self.out))
        return self.out[rf]

    #Video verification function (input video, uses feature as reference)
    def video_verification(self, vid = vid_config, save_vid = vid_save_loc , featureRs = [], subject = None):
        if subject:
            self.featureRs = self.get_feature_bank(subject = subject)
        else:
            self.featureRs = featureRs
        self.zoomvid = vid
        save_vid = save_vid
        face_save = face_save_loc
        cap = cv2.VideoCapture(vid)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (self.w,self.h))
        writer2 = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (self.w,self.h))
        self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h            
        self.is_anchor = False
        facial5points = [0,0,0,0,0,0,0,0,0,0]
        landmarks = [0,0,0,0,0,0,0,0,0,0]
        box = [0, 0, 0, 0]
        
        i = 0
        label = 'No Mask'
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                # try:
                start_time = time.time()

                is_face = False

                self.img = frame
                self.img = np.copy(frame[:, :, ::-1])

                picked_boxes, picked_landmarks, picked_scores = self.face_detector()

                if self.is_anchor == False:
                    is_face, box[0], box[1], box[2], box[3], box0, box1, box2, box3, landmark = self.compute_anchor(picked_boxes, picked_landmarks, picked_scores)
 
                elif self.is_anchor == True:
                    is_face, box[0], box[1], box[2], box[3], box0, box1, box2, box3, landmark = self.return_anchor(picked_boxes, picked_landmarks, picked_scores)


                if is_face == True:
                    landmark = landmark.detach().cpu()

                    for l in range(10):
                        landmarks[l] = int(landmark[l].item())
                    output_size = (112,112)
                    facial5points = [landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8], landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]]

                    facial5points = np.reshape(facial5points, (2, 5))

                    default_square = True
                    inner_padding_factor = 0.25
                    outer_padding = (0, 0)

                    # get the reference 5 landmarks position in the crop settings
                    reference_5pts = get_reference_facial_points(
                        output_size, inner_padding_factor, outer_padding, default_square)

                    # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
                    dst_img = warp_and_crop_face(self.img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
                    cv2.imwrite(alignment_save, dst_img)
                    # img = cv.resize(raw, (224, 224))
                    # cv.imwrite('images/{}_img.jpg'.format(i), img)
                    

                    face = frame[box[1]:box[3], box[0]:box[2]]
                    cv2.imwrite(face_save, face)


                    prediction = 1

                    prediction = self.mask_detection(face)
                
                    label = "Mask" if prediction == 0 else "No Mask"

                    if label == 'Mask':
                        pass
                    else:
                        if self.valface > time_to_verify:
                            face = alignment_save
                            start_time = time.time()
                            self.featureLs = self.verify(face)
                            self.score = self.compare_features()
                            
                            result = self.verify_result(self.score, self.threshold)
                            self.valface = 1

                            det = 1
                            if result == 'SAME':
                                self.flag = 1
                            else:
                                self.flag = 0
                                self.is_anchor = False
                                self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h            
                        # print('score: ', self.score)

                            # print('__________________________________________________________________')
                            # print()


                end_time = time.time()
                # print('time:  ', end_time - start_time)
                self.time_taken.append((end_time - start_time)) #To validate time


                self.img = self.draw(box, self.flag, box0, box1, box2, box3, label)

                self.start += 1
                self.valface += 1
                writer.write(self.img)
                writer2.write(frame)
                cv2.imshow('RetinaFace',self.img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                i += 1
            
                # except:
                #     print('passed')
                #     pass
            else:
                break
        

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

    #Return features from a single image
    def image_feature_extractor(self, image_loc = left_image_loc, face_loc = left_face_loc):
        box = [0,0,0,0]
        image_loc = image_loc
        self.img = cv2.imread(image_loc)

        self.img = np.copy(self.img[:, :, ::-1])
        self.h, self.w, c = self.img.shape
        self.vbox0, self.vbox1, self.vbox2, self.vbox3 = 0, 0, self.w, self.h            

        is_face, box[0], box[1], box[2], box[3], box0, box1, box2, box3, landmarks  = self.image_face_detector()
        output_size = (112,112)
        facial5points = [landmarks[0], landmarks[2], landmarks[4], landmarks[6], landmarks[8], landmarks[1], landmarks[3], landmarks[5], landmarks[7], landmarks[9]]
        # detector = RetinafaceDetector()
        # landmarks , facial5points = detector.detect_faces(self.img)

        # print(landmarks)
        # exit()
        # print(facial5points)
        # exit()
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
        
        return self.verify(face_loc)

    #Image verification, input 2 images
    def image_verification(self, imageL_loc = left_image_loc, imageR_loc = right_image_loc, faceL_loc = left_face_loc, faceR_loc = right_face_loc):
        
        self.featureLs = fv.image_feature_extractor(imageL_loc, faceL_loc)
        self.featureRs = fv.image_feature_extractor(imageR_loc, faceR_loc)

        self.score = self.compare_features()
        
        result = self.verify_result(self.score, self.threshold)
        
        print(self.score, result)

    #Input output function
    def verification(self):
        pass

if __name__=='__main__':


    # fv = face_verifier(model_type = '18')
    fv = face_verifier(model_type)

    # summary(fv.model, (3,112,112))
    # exit()

    #Vid = 0 for live webcam input, else put in video location
    vid = 0
    # featureL = fv.image_feature_extractor()
    featureR = fv.image_feature_extractor(right_image_loc, right_face_loc)
    
    # fv.image_feature_extractor(image_loc='data/a.jpg')

    fv.video_verification(vid = 0, featureRs = featureR)
    # fv.video_verification(subject = cur_subject)
    # fv.image_verification()


    featureL = fv.image_feature_extractor()
    featureR = fv.image_feature_extractor(right_image_loc, right_face_loc)

    score = fv.compare_features(featureL, featureR)
    result = fv.verify_result(score, 0.4)
    print(score, result)

    
