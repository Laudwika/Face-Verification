import os

alignment_save = 'test.jpg'
face_save_loc = 'data/ref/ref.jpg'
model18 = 'student-model.h5'
model101 = 'model101.h5'
mask_model = 'MaskModelBest.h5'
feature_path_loc = './results/cur_epoch_test_result.mat'
face_bank_loc = 'data/features/'
video_output = 'data/zout/clean_test.mp4'
model_type = '18'
right_image_loc = 'data/a.jpg'
right_face_loc = 'data/ac.jpg'
left_image_loc = 'data/L,jpg'
left_face_loc = 'data/Lc.jpg'
vid_config = 0
vid_save_loc = 'data/zout/test.mp4'
cur_subject = 's0'
box_constant = 1.5