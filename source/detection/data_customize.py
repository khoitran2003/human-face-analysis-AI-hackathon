import argparse
import os
import shutil
import pandas as pd
import cv2
import ast


def get_args():
    parser = argparse.ArgumentParser("Yolo Custom Dataset")
    parser.add_argument("--root", "-r", type=str, default="offical_data")
    parser.add_argument("--det_path", "-d", type=str, default="det_data")
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
    print('Bitwise: ', len(file_name_bitwise))

    # check det dir exist
    if os.path.isdir(args.det_path):
        shutil.rmtree(args.det_path)
    os.makedirs(args.det_path)
    os.makedirs(os.path.join(args.det_path, "images"))
    os.makedirs(os.path.join(args.det_path, "images", "train"))
    os.makedirs(os.path.join(args.det_path, "images", "val"))
    os.makedirs(os.path.join(args.det_path, "labels"))
    os.makedirs(os.path.join(args.det_path, "labels", "train"))
    os.makedirs(os.path.join(args.det_path, "labels", "val"))

    # check clf dir exist
    if os.path.isdir(args.clf_path):
        shutil.rmtree(args.clf_path)
    os.makedirs(args.clf_path)
    os.makedirs(os.path.join(args.clf_path, "train"))
    os.makedirs(os.path.join(args.clf_path, "val"))

    for image_id, path in enumerate(file_name_bitwise):

        # verify bbox 
        new_name_image = "{}.jpg".format(image_id + 1)
        current_person = labels_csv[labels_csv["file_name"] == path]
        current_person = current_person["bbox"].values.tolist()

        current_face = [ast.literal_eval(item) for item in current_person]
        current_face = [[float(item) for item in sublist] for sublist in current_face]

        # read image
        image = cv2.imread(os.path.join(args.root, 'images', path))
        width_image = image.shape[1]
        height_image = image.shape[0]

        # check: draw bbox on image 
        # for face in current_person:
        #     xmin, ymin, width, height = face
        #     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmin + width), int(ymin + height)), (0, 0, 255), 2)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        print(image_id + 1)

        # Change file_name CSV file
        # for i, test_str in enumerate(current_person):
        #     index = labels_csv[labels_csv['bbox'] == test_str].index.values[0]
        #     labels_csv.at[index, 'file_name'] = f'{image_id + 1}_{i + 1}.jpg'

        # Make train, val det data
        if image_id < int(len(file_name_bitwise) * 0.9):

            # det
            shutil.copyfile(
                os.path.join(args.root, "images", path),
                os.path.join(args.det_path, "images", "train", new_name_image),
            )
            with open(
                    os.path.join(
                        args.det_path, "labels", "train", "{}.txt".format(image_id + 1)
                    ),
                    "w",
            ) as text_file:
                for i, face in enumerate(current_face):
                    xmin, ymin, width, height = face
                    xcent_n = (xmin + width / 2) / width_image
                    ycent_n = (ymin + height / 2) / height_image
                    w_n = width / width_image
                    h_n = height / height_image

                    # det txt
                    text_file.write(
                        "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(xcent_n, ycent_n, w_n, h_n)
                    )

                    # clf jpg
                    cropped_image = image[int(ymin):int(ymin + height), int(xmin):int(xmin + width)]
                    cv2.imwrite(os.path.join(args.clf_path, 'train', '{}_{}.jpg'.format(image_id + 1, i + 1)),
                                cropped_image)

        else:
            shutil.copyfile(
                os.path.join(args.root, "images", path),
                os.path.join(args.det_path, "images", "val", new_name_image),
            )
            with open(
                    os.path.join(
                        args.det_path, "labels", "val", "{}.txt".format(image_id + 1)
                    ),
                    "w",
            ) as text_file:
                for i, face in enumerate(current_face):
                    xmin, ymin, width, height = face
                    xcent_n = (xmin + width / 2) / width_image
                    ycent_n = (ymin + height / 2) / height_image
                    w_n = width / width_image
                    h_n = height / height_image

                    # det txt
                    text_file.write(
                        "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(xcent_n, ycent_n, w_n, h_n)
                    )

                    # clf jpg
                    cropped_image = image[int(ymin):int(ymin + height), int(xmin):int(xmin + width)]
                    cv2.imwrite(os.path.join(args.clf_path, 'val', '{}_{}.jpg'.format(image_id + 1, i + 1)),
                                cropped_image)

    # labels_csv.drop(['Unnamed: 0'], axis=1)
    # labels_csv.to_csv('clf_data/all_labels.csv', index=False)





if __name__ == "__main__":
    args = get_args()
    main(args)
