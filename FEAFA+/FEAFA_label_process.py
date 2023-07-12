import os
import numpy as np
import pandas as pd

# Images should be face aligned and cropped and stored in the images_path
#   directory in the following format: '{images_path}/{subject}_{frame}.jpg'
image_path = "/home/kyle/school/farapy/FEAFA+/FEAFA-A/processed/"
list_path_prefix = "../../FEAFA+/FEAFA-A/list/"
label_path_prefix = "../../FEAFA+/FEAFA-A/"

# Define data split into 3 parts
part1 = [
    "PV008",
    "PV009",
    "PV010",
    "PV011",
    "PV013",
    "PV019",
    "PV020",
    "PV024",
    "PV029",
    "PV030",
    "PV031",
    "PV033",
    "PV039",
    "PV040",
    "PV046",
    "PV053",
    "PV054",
    "PV055",
    "PV059",
    "PV060",
    "PV062",
    "PV076",
    "PV078",
    "PV082",
    "PV089",
    "PV100",
    "PV101",
    "PV102",
    "PV103",
    "PV107",
    "PV109",
    "PV110",
    "PV114",
    "PV115",
    "PV116",
    "PV117",
    "PV119",
    "PV120",
    "PV122",
    "PV123",
    "PV124",
    "PV125",
]
part2 = [
    "PV003",
    "PV004",
    "PV005",
    "PV007",
    "PV012",
    "PV014",
    "PV015",
    "PV016",
    "PV018",
    "PV021",
    "PV022",
    "PV023",
    "PV025",
    "PV032",
    "PV037",
    "PV041",
    "PV044",
    "PV045",
    "PV050",
    "PV051",
    "PV052",
    "PV057",
    "PV067",
    "PV068",
    "PV070",
    "PV071",
    "PV072",
    "PV074",
    "PV080",
    "PV085",
    "PV086",
    "PV087",
    "PV088",
    "PV090",
    "PV091",
    "PV094",
    "PV095",
    "PV097",
    "PV098",
    "PV111",
    "PV126",
    "PV127",
]
part3 = [
    "PV001",
    "PV002",
    "PV006",
    "PV017",
    "PV026",
    "PV027",
    "PV028",
    "PV034",
    "PV035",
    "PV036",
    "PV038",
    "PV042",
    "PV043",
    "PV047",
    "PV048",
    "PV049",
    "PV056",
    "PV058",
    "PV061",
    "PV063",
    "PV064",
    "PV065",
    "PV066",
    "PV069",
    "PV073",
    "PV075",
    "PV077",
    "PV079",
    "PV081",
    "PV083",
    "PV084",
    "PV092",
    "PV093",
    "PV096",
    "PV099",
    "PV104",
    "PV105",
    "PV106",
    "PV108",
    "PV112",
    "PV113",
    "PV118",
    "PV121",
]

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

frame_list = {1: [], 2: [], 3: []}
numpy_list = {1: None, 2: None, 3: None}
part1_subjects = set(part1)
part2_subjects = set(part2)

# Create/clear test fold image path files
with open(list_path_prefix + "FEAFA_test_img_path_fold1.txt", "w") as f:
    u = 0
with open(list_path_prefix + "FEAFA_test_img_path_fold2.txt", "w") as f:
    u = 0
with open(list_path_prefix + "FEAFA_test_img_path_fold3.txt", "w") as f:
    u = 0

# Iterate through all images, add to coresponding frame_list, save test fold image path
data_dir = os.fsencode(image_path)
for file in sorted(os.listdir(data_dir)):
    filename = os.fsdecode(file)
    subject = filename[:5]

    part = 1 if subject in part1_subjects else 2 if subject in part2_subjects else 3
    test_fold = 1 if part == 3 else 2 if part == 2 else 3
    
    filename = image_path + filename
    frame_list[part].append(filename)

    with open(list_path_prefix + f"FEAFA_test_img_path_fold{test_fold}.txt", "a+") as f:
        f.write(filename + "\n")

# Get labels for each frame in each part, add to corresponding numpy_list, save test fold labels
for part in [1, 2, 3]:
    test_fold = 1 if part == 3 else 2 if part == 2 else 3
    numpy_list[part] = np.zeros((len(frame_list[part]), 24), dtype=float)
    for j, imageName in enumerate(frame_list[part]):
        subject = imageName[-14:-9]
        frame = imageName[-8:-4]

        label_path = label_path_prefix + subject + ".output/" + f"{frame:0>8}" + ".auw"
        with open(label_path, "r") as label_file:
            labels = label_file.readline().split()

        for i, label in enumerate(labels):
            numpy_list[part][j][i] = float(label)

    np.savetxt(
        list_path_prefix + f"FEAFA_test_label_fold{test_fold}.txt",
        numpy_list[part],
        fmt="%f",
        delimiter=" ",
    )

# Create/clear train fold image path files
with open(list_path_prefix + "FEAFA_train_img_path_fold1.txt", "w") as f:
    u = 0
with open(list_path_prefix + "FEAFA_train_img_path_fold2.txt", "w") as f:
    u = 0
with open(list_path_prefix + "FEAFA_train_img_path_fold3.txt", "w") as f:
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
        with open(list_path_prefix + f"FEAFA_train_img_path_fold{fold}.txt", "a+") as f:
            f.write(frame + "\n")

    np.savetxt(
        list_path_prefix + f"FEAFA_train_label_fold{fold}.txt",
        train_label_list,
        fmt="%f",
    )
