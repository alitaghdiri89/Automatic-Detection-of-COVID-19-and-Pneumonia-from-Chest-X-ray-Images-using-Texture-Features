from skimage.feature import graycomatrix, graycoprops
import numpy as np
import constants as const

# parameter image is a 2D numpy array
# returns a dictionary of the features:
#   {'main_contrast': value,
#    'main_correlation': value,
#    'main_ASM': value,
#    'main_homogeneity': value,
#    'diff_contrast': value,
#    'diff_correlation': value,
#    'diff_ASM': value,
#    'diff_homogeneity': value
#    'mean_contrast': value,
#    'mean_correlation': value,
#    'mean_ASM': value,
#    'mean_homogeneity': value}


def extract_features(image, mean_normal_image, class_count):
    features = dict()
    dist_from_mean = get_distance_from_mean(image, mean_normal_image)
    calc_features(dist_from_mean, features, class_count, 'mean')
    image //= const.COMPRESSION_FACTOR
    diff_image = calc_diff_image(image)
    calc_features(image, features, class_count, 'main')
    calc_features(diff_image, features, class_count, 'diff')
    return features


def get_distance_from_mean(image, mean_normal_image):
    difference = np.subtract(image, mean_normal_image)
    difference //= const.COMPRESSION_FACTOR
    return difference


def calc_diff_image(image):
    diff_image = np.diff(image.astype(np.int8))
    diff_image = np.absolute(diff_image)
    return diff_image.astype(np.uint8)


def calc_features(image, features, class_count, prefix):
    glcm = graycomatrix(image, const.DISTANCE, const.ANGLES[class_count], normed=True)
    for feature_name in const.FEATURE_NAMES:
        features[f'{prefix}_{feature_name}'] = graycoprops(glcm, prop=feature_name)[0][0]
