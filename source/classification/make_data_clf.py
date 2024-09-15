import argparse
import os
import shutil
import pandas as pd
import cv2
import ast


def get_args():
    parser = argparse.ArgumentParser("Yolo Custom Dataset")
    parser.add_argument("--root", "-r", type=str, default="offical_data")
    parser.add_argument("--clf_path", "-c", type=str, default="clf_data")
    return parser.parse_args()


def main(args):
    image_folder = os.path.join(args.root, "images")
    images = os.listdir(image_folder)
    # print(len(images))
    labels_csv = pd.read_csv(os.path.join(args.root, "labels/all_labels.csv"))
    file_name = labels_csv["file_name"].to_list()
    # print(len(file_name))
    file_name_bitwise = set(images) & set(file_name)
    # print('Bitwise: ', len(file_name_bitwise))

    # check clf dir exist
    if os.path.isdir(args.clf_path):
        shutil.rmtree(args.clf_path)
    os.makedirs(args.clf_path)
    os.makedirs(os.path.join(args.clf_path, "train"))
    os.makedirs(os.path.join(args.clf_path, "val"))

    # current_bbox = [ast.literal_eval(bbox) for bbox in labels_csv['bbox'].values.tolist()]
    # current_bbox = [[float(item) for item in sublist] for sublist in current_bbox]
    # print(len(current_bbox))

    current_image = [image_name for image_name in labels_csv['file_name'].values.tolist()]

    i = 0
    for image_name, bbox in zip(current_image, labels_csv['bbox'].values.tolist()):
        print(i+1)
        ori_bbox = ast.literal_eval(bbox)
        current_bbox = [float(item) for item in ori_bbox]

        image = cv2.imread(os.path.join(args.root, 'images', image_name))

        xmin, ymin, width, height = current_bbox
        cropped_image = image[int(ymin):int(ymin + height), int(xmin):int(xmin + width), :]

        index = labels_csv[labels_csv['bbox'] == bbox].index.values[0]
        labels_csv.at[index, 'file_name'] = f'{i + 1}.jpg'

        if i + 1 < int(len(current_image)*0.9):
            cv2.imwrite(os.path.join(args.clf_path, 'train', '{}.jpg'.format(i + 1)),
                        cropped_image)

        else:
            cv2.imwrite(os.path.join(args.clf_path, 'val', '{}.jpg'.format(i + 1)),
                        cropped_image)

        i += 1

    labels_csv.to_csv(os.path.join(args.clf_path, 'all_labels.csv'), index=False)

if __name__ == "__main__":
    args = get_args()
    main(args)
