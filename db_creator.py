# Created by Ali on 05-Aug-21
import numpy as np
import constants as const
import pandas as pd
from os import path, listdir, mkdir
from skimage import io
from skimage.util import img_as_ubyte
from feature_extractor import extract_features
from tqdm import tqdm  # to show a progress bar
import cv2


def get_image(directory, image_name):
    image_path = path.join(directory, image_name)
    image_path = image_path.replace('\\', '\\\\')
    image = io.imread(image_path, as_gray=True)
    image = img_as_ubyte(image)
    image = preprocess(image)
    return image


def add_to_feature_data(feature_data, features, label):
    features['class'] = label
    for feature_name, feature_value in features.items():
        feature_data[feature_name] = feature_data.get(feature_name, list()) + [feature_value]


def get_db_file_path(class_count):
    file_name = f'{class_count}_classes.csv'
    db_path = path.join(const.DB_DIRECTORY, file_name)
    return db_path


def create_db_files(feature_data, class_count):
    df = pd.DataFrame(feature_data)
    while True:
        try:
            df.to_csv(get_db_file_path(class_count), index=False)
            break
        except FileNotFoundError:
            mkdir(const.DB_DIRECTORY)


def preprocess(image):
    image = cv2.resize(image, const.PATCH_DIM, interpolation=cv2.INTER_AREA)
    margin_size = const.PATCH_DIM[1] // 10
    image = image[margin_size:image.shape[0] - margin_size, margin_size:image.shape[1] - margin_size]
    return image


def create_class_mean_image(file_name, class_name):
    images_sum = None
    class_directory = path.join(const.IMAGES_DIRECTORY, class_name)
    image_name_list = listdir(class_directory)
    image_count = len(image_name_list)
    print(f"creating {file_name}")
    for image_name in tqdm(image_name_list):
        image = get_image(class_directory, image_name)
        if images_sum is None:
            images_sum = np.zeros(image.shape)
        images_sum += image
    mean_image = (images_sum // image_count).astype(np.uint8)
    io.imsave(file_name, mean_image)
    return mean_image


def get_class_mean_image(class_name):
    file_name = f'mean_image_{class_name}.png'
    try:
        mean_image = io.imread(file_name, as_gray=True)
    except FileNotFoundError:
        mean_image = create_class_mean_image(file_name, class_name)
    return mean_image


def create_db(class_count):
    feature_data = dict()
    mean_normal_image = get_class_mean_image("normal")
    for i, class_name in enumerate(const.CLASS_SETS[f'Cohen-Kermani {class_count} classes']):
        directory = path.join(const.IMAGES_DIRECTORY, class_name)
        print(f'Class: {class_name}')
        image_name_list = listdir(directory)
        for image_name in tqdm(image_name_list):
            image = get_image(directory, image_name)
            features = extract_features(image, mean_normal_image, class_count)
            add_to_feature_data(feature_data, features, label=i)
    create_db_files(feature_data, class_count)
