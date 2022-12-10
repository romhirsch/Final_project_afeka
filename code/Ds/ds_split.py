import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
import shutil

def split_dataset_test_train(directory, target_path_test, target_path_train, percent=0.2):
    if not os.path.exists(directory):
        return
    sub_dirs = os.listdir(directory)
    images = []
    labels = []
    for sub_dir in sub_dirs:
        abs_path = os.path.join(directory, sub_dir)
        image_list = os.listdir(abs_path)  # list of all image names in the directory
        image_list = list(map(lambda x: os.path.join(abs_path, x), image_list))
        images.extend(image_list)
        labels.extend([sub_dir] * len(image_list))
    df = pd.DataFrame({"Images": images, "Labels": labels})
    if not os.path.exists(target_path_test):
        os.mkdir(target_path_test)
    if not os.path.exists(target_path_train):
        os.mkdir(target_path_train)
    for label in df['Labels'].unique():
        if not os.path.exists(os.path.join(target_path_test, label)):
            os.mkdir(os.path.join(target_path_test, label))
        if not os.path.exists(os.path.join(target_path_train, label)):
            os.mkdir(os.path.join(target_path_train, label))
        sub_df = df[df['Labels'] == label].copy()
        sub_df = shuffle(sub_df)
        sub_df.reset_index(inplace=True)
        end_test = int(sub_df.shape[0]*percent)
        for path in sub_df.loc[:end_test, 'Images']:
            file_name = os.path.basename(path)
            label_path = os.path.join(target_path_test, label)
            target_path = os.path.join(label_path, file_name)
            shutil.copyfile(path, target_path)
        for path in sub_df.loc[end_test:, 'Images']:
            file_name = os.path.basename(path)
            label_path = os.path.join(target_path_train, label)
            target_path = os.path.join(label_path, file_name)
            shutil.copyfile(path, target_path)

if __name__ == '__main__':
    directory = r"E:\imagenet_orig"
    target_path_test = r"E:\imagenet_test"
    target_path_train = r"E:\imagenet_train"
    split_dataset_test_train(directory, target_path_test, target_path_train, percent=0.2)