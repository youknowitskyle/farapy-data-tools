import os
import numpy as np
import pandas as pd

# You nead downloading DISFA including 'ActionUnit_Labels'
label_path = '../data/DISFA/ActionUnit_Labels'
list_path_prefix = '../data/DISFA/list/'
images_path = '../data/DISFA/img'

part1 = ['PV008', 'PV009', 'PV010', 'PV011', 'PV013', 'PV019', 'PV020', 'PV024', 'PV029', 'PV030', 'PV031', 'PV033', 'PV039', 'PV040', 'PV046', 'PV053', 'PV054', 'PV055', 'PV059', 'PV060', 'PV062', 'PV076', 'PV078', 'PV082', 'PV089', 'PV100', 'PV101', 'PV102', 'PV103', 'PV107', 'PV109', 'PV110', 'PV114', 'PV115', 'PV116', 'PV117', 'PV119', 'PV120', 'PV122', 'PV123', 'PV124', 'PV125', 'PV126'] 
part2 = ['PV000', 'PV003', 'PV004', 'PV005', 'PV007', 'PV012', 'PV014', 'PV015', 'PV016', 'PV018', 'PV021', 'PV022', 'PV023', 'PV025', 'PV032', 'PV037', 'PV041', 'PV044', 'PV045', 'PV050', 'PV051', 'PV052', 'PV057', 'PV067', 'PV068', 'PV070', 'PV071', 'PV072', 'PV074', 'PV080', 'PV085', 'PV086', 'PV087', 'PV088', 'PV090', 'PV091', 'PV094', 'PV095', 'PV097', 'PV098', 'PV111', 'PV127'] 
part3 = ['PV001', 'PV002', 'PV006', 'PV017', 'PV026', 'PV027', 'PV028', 'PV034', 'PV035', 'PV036', 'PV038', 'PV042', 'PV043', 'PV047', 'PV048', 'PV049', 'PV056', 'PV058', 'PV061', 'PV063', 'PV064', 'PV065', 'PV066', 'PV069', 'PV073', 'PV075', 'PV077', 'PV079', 'PV081', 'PV083', 'PV084', 'PV092', 'PV093', 'PV096', 'PV099', 'PV104', 'PV105', 'PV106', 'PV108', 'PV112', 'PV113', 'PV118', 'PV121']

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

au_idx = [1, 2, 4, 6, 9, 12, 25, 26]


with open(list_path_prefix + 'DISFA_test_img_path_fold3.txt', 'w') as f:
    u = 0
# loop through subjects
    # loop through frames in image directory, count the amount of valid frames, store valid frame numbers in a set, store frame names in fold.txt file, add to frame list
    # construct an array [frame, au number] = au intensity
    # loop through au number
    # loop through each au number's corresponding label file
    # if frame is in set of valid frame numbers
    # add to numpy array
    # append numpy label array to part1_numpy_list
# change part1_numpy_list to an np array
# write part1_numpy_list to test_label_fold.txt

part1_frame_list = []
part1_numpy_list = []
for fr in part1:
    fr_path = os.path.join(label_path, fr)
    fr_directory = os.fsencode(f'{images_path}/{fr}/')
    total_frame = 0
    valid_frames = set()
    for file in sorted(os.listdir(fr_directory)):
        filename = os.fsdecode(file)
        if "left" in filename:
            total_frame += 1
            valid_frames.add(int(filename[-8:-4]))
            frame_img_name = fr + '/' + filename
            part1_frame_list.append(frame_img_name)
            with open(list_path_prefix + 'DISFA_test_img_path_fold3.txt', 'a+') as f:
                f.write(frame_img_name+'\n')
    au_label_array = np.zeros((total_frame, 8), dtype=int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path, fr+'_au'+str(au) + '.txt')
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if frameIdx in valid_frames:
                    au_label_array[t, ai] = AUIntensity
    part1_numpy_list.append(au_label_array)
part1_numpy_list = np.concatenate(part1_numpy_list, axis=0)
# # part1 test for fold3
np.savetxt(list_path_prefix + 'DISFA_test_label_fold3.txt',
           part1_numpy_list, fmt='%d', delimiter=' ')


#################################################################################

with open(list_path_prefix + 'DISFA_test_img_path_fold2.txt', 'w') as f:
    u = 0

part2_frame_list = []
part2_numpy_list = []
for fr in part2:
    fr_path = os.path.join(label_path, fr)
    fr_directory = os.fsencode(f'{images_path}/{fr}/')
    total_frame = 0
    valid_frames = set()
    for file in sorted(os.listdir(fr_directory)):
        filename = os.fsdecode(file)
        if "left" in filename:
            total_frame += 1
            valid_frames.add(int(filename[-8:-4]))
            frame_img_name = fr + '/' + filename
            part2_frame_list.append(frame_img_name)
            with open(list_path_prefix + 'DISFA_test_img_path_fold2.txt', 'a+') as f:
                f.write(frame_img_name+'\n')
    au_label_array = np.zeros((total_frame, 8), dtype=int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path, fr+'_au'+str(au) + '.txt')
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if frameIdx in valid_frames:
                    au_label_array[t, ai] = AUIntensity
    part2_numpy_list.append(au_label_array)
part2_numpy_list = np.concatenate(part2_numpy_list, axis=0)
# # part2 test for fold2
np.savetxt(list_path_prefix + 'DISFA_test_label_fold2.txt',
           part2_numpy_list, fmt='%d', delimiter=' ')


#################################################################################
with open(list_path_prefix + 'DISFA_test_img_path_fold1.txt', 'w') as f:
    u = 0

part3_frame_list = []
part3_numpy_list = []
for fr in part3:
    fr_path = os.path.join(label_path, fr)
    fr_directory = os.fsencode(f'{images_path}/{fr}/')
    total_frame = 0
    valid_frames = set()
    for file in sorted(os.listdir(fr_directory)):
        filename = os.fsdecode(file)
        if "left" in filename:
            total_frame += 1
            valid_frames.add(int(filename[-8:-4]))
            frame_img_name = fr + '/' + filename
            part3_frame_list.append(frame_img_name)
            with open(list_path_prefix + 'DISFA_test_img_path_fold1.txt', 'a+') as f:
                f.write(frame_img_name+'\n')
    au_label_array = np.zeros((total_frame, 8), dtype=int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path, fr+'_au'+str(au) + '.txt')
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if frameIdx in valid_frames:
                    au_label_array[t, ai] = AUIntensity
    part3_numpy_list.append(au_label_array)
part3_numpy_list = np.concatenate(part3_numpy_list, axis=0)
# # part3 test for fold1
np.savetxt(list_path_prefix + 'DISFA_test_label_fold1.txt',
           part3_numpy_list, fmt='%d', delimiter=' ')


#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold1.txt', 'w') as f:
    u = 0
train_img_label_fold1_list = part1_frame_list + part2_frame_list
for frame_img_name in train_img_label_fold1_list:
    with open(list_path_prefix + 'DISFA_train_img_path_fold1.txt', 'a+') as f:
        f.write(frame_img_name + '\n')
train_img_label_fold1_numpy_list = np.concatenate(
    (part1_numpy_list, part2_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold1.txt',
           train_img_label_fold1_numpy_list, fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold2.txt', 'w') as f:
    u = 0
train_img_label_fold2_list = part1_frame_list + part3_frame_list
for frame_img_name in train_img_label_fold2_list:
    with open(list_path_prefix + 'DISFA_train_img_path_fold2.txt', 'a+') as f:
        f.write(frame_img_name + '\n')
train_img_label_fold2_numpy_list = np.concatenate(
    (part1_numpy_list, part3_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold2.txt',
           train_img_label_fold2_numpy_list, fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold3.txt', 'w') as f:
    u = 0
train_img_label_fold3_list = part2_frame_list + part3_frame_list
for frame_img_name in train_img_label_fold3_list:
    with open(list_path_prefix + 'DISFA_train_img_path_fold3.txt', 'a+') as f:
        f.write(frame_img_name + '\n')
train_img_label_fold3_numpy_list = np.concatenate(
    (part2_numpy_list, part3_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold3.txt',
           train_img_label_fold3_numpy_list, fmt='%d')
