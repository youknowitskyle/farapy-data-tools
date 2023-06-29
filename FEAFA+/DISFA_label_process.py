import os
import numpy as np
import pandas as pd

# Images should be face aligned and cropped and stored in the images_path
#   directory in the following format: '{images_path}/{subject}/{frame}.jpg'
list_path_prefix = "../../FEAFA+/FEAFA-B/list/"
label_path_prefix = "../../FEAFA+/FEAFA-B/"
images_path = "../../DISFA/processed"

# Define  data split into 3 parts
parts = {
    1: [
        "SN002",
        "SN010",
        "SN001",
        "SN026",
        "SN027",
        "SN032",
        "SN030",
        "SN009",
        "SN016",
    ],
    2: [
        "SN013",
        "SN018",
        "SN011",
        "SN028",
        "SN012",
        "SN006",
        "SN031",
        "SN021",
        "SN024",
    ],
    3: [
        "SN003",
        "SN029",
        "SN023",
        "SN025",
        "SN008",
        "SN005",
        "SN007",
        "SN017",
        "SN004",
    ],
}

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part

# Create/clear test fold image path files
with open(list_path_prefix + "DISFA_test_img_path_fold3.txt", "w") as f:
    u = 0
with open(list_path_prefix + "DISFA_test_img_path_fold2.txt", "w") as f:
    u = 0
with open(list_path_prefix + "DISFA_test_img_path_fold1.txt", "w") as f:
    u = 0

frame_list = {1: [], 2: [], 3: []}
numpy_list = {1: None, 2: None, 3: None}

# Iterate through all images, add to coresponding frame_list, save test fold image path
for part in [1, 2, 3]:
    test_fold = 1 if part == 3 else 2 if part == 2 else 3
    for subject in parts[part]:
        sub_dir = os.fsencode(f"{images_path}/{subject}/")

        for file in sorted(os.listdir(sub_dir)):
            filename = os.fsdecode(file)
            if "full" in filename:
                frame_path = f"{subject}/{filename}"
                frame_list[part].append(frame_path)
                with open(
                    f"{list_path_prefix}DISFA_test_img_path_fold{test_fold}.txt", "a+"
                ) as f:
                    f.write(frame_path + "\n")

# Get labels for each frame in each part, add to corresponding numpy_list, save test fold labels
for part in [1, 2, 3]:
    test_fold = 1 if part == 3 else 2 if part == 2 else 3
    numpy_list[part] = np.zeros((len(frame_list[part])), dtype=float)
    for j, imagePath in enumerate(frame_list[part]):
        subject = imagePath[:5]
        frame = imagePath[6:10]

        label_path = f"{label_path_prefix}{subject}.output/{frame:0>8}.auw"
        with open(label_path, "r") as label_file:
            labels = label_file.readline().split()

        for i, label in enumerate(labels):
            numpy_list[part][j][i] = float(label)

    np.savetxt(
        f"{list_path_prefix}DISFA_test_label_fold{test_fold}.txt",
        numpy_list[part],
        fmt="%f",
        delimiter=" ",
    )

# Create/clear train fold image path files
with open(list_path_prefix + "DISFA_train_img_path_fold1.txt", "w") as f:
    u = 0
with open(list_path_prefix + "DISFA_train_img_path_fold2.txt", "w") as f:
    u = 0
with open(list_path_prefix + "DISFA_train_img_path_fold3.txt", "w") as f:
    u = 0

# Build image path and label lists for each train fold and save to file
for fold in [1, 2, 3]:
    if fold == 1:
        train_img_list = frame_list[1] + frame_list[2]
        train_label_list = np.concatenate((numpy_list[1], numpy_list[2]), axis=0)
    elif fold == 2:
        train_img_list = frame_list[1] + frame_list[3]
        train_label_list = np.concatenate((numpy_list[1], numpy_list[3]), axis=0)
    else:
        train_img_list = frame_list[2] + frame_list[3]
        train_label_list = np.concatenate((numpy_list[2], numpy_list[3]), axis=0)

    for frame in train_img_list:
        with open(f"{list_path_prefix}DISFA_train_img_path_fold{fold}.txt", "a+") as f:
            f.write(frame + "\n")

    np.savetxt(
        f"{list_path_prefix}DISFA_train_label_fold{fold}.txt",
        train_label_list,
        fmt="%f",
    )
