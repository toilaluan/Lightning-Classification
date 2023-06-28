from glob import glob
import argparse
import pandas as pd
import os
from clearml import Dataset
from random import shuffle

LABEL_TO_IDX = {
    "Lilly": 0,
    "Lotus": 1,
    "Orchid": 2,
    "Sunflower": 3,
    "Tulip": 4,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_project", type=str)
    parser.add_argument("--test_ratio", type=float)
    args = parser.parse_args()
    assert args.test_ratio >= 0 and args.test_ratio <= 1
    return args


def get_all_image_paths(root_folder):
    return glob(root_folder + "/**/*")


def prepare_label(image_paths):
    labels = []
    for image_path in image_paths:
        label = image_path.split("/")[-2]
        labels.append(label)
    return labels


def data_to_clearml(args, image_paths):
    ds = Dataset.create(
        dataset_name=args.dataset_name, dataset_project=args.dataset_project
    )
    ds.add_files(image_paths, dataset_path="images")
    ds.add_files(os.path.join(args.root_folder), "annotation.csv")
    ds.upload()
    ds.finalize(True)


def train_test_split(data_csv, args):
    total_data_len = len(data_csv)
    test_len = int(total_data_len * args.test_ratio)
    train_len = total_data_len - test_len
    is_train = [1] * train_len
    is_test = [0] * test_len
    is_train = [*is_train, *is_test]
    shuffle(is_train)
    data_csv["is_train"] = is_train
    return data_csv


if __name__ == "__main__":
    args = get_args()
    image_paths = get_all_image_paths(args.root_folder)
    image_names = [x.split("/")[-1] for x in image_paths]
    labels = prepare_label(image_paths)
    ground_truths = [LABEL_TO_IDX[x] for x in labels]
    data_csv = pd.DataFrame(
        {"image_name": image_names, "label": labels, "ground_truth": ground_truths}
    )
    data_csv = train_test_split(data_csv=data_csv, args=args)
    data_csv.to_csv(os.path.join(args.root_folder, "annotation.csv"))
    data_to_clearml(image_paths=image_paths, args=args)
