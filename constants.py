from os import path
import numpy as np

ANGLES = {2: [0], 3: [np.pi / 2]}
DISTANCE = [1]

TRAIN_SIZE = 0.8
DB_DIRECTORY = path.join('.', 'feature_databases')
PATCH_DIM = (1300, 1300)
FEATURE_NAMES = ['contrast', 'correlation', 'ASM', 'homogeneity']  # ASM is called energy in the paper
IMAGES_DIRECTORY = path.join('.', 'images')
NUMBER_OF_EXPERIMENTS = 150
COHEN_KERMANI_2_CLASSES = ('normal', 'covid')  # section 4.3
COHEN_KERMANI_3_CLASSES = COHEN_KERMANI_2_CLASSES + ('pneumonia',)  # section 4.4
CLASS_SETS = {'Cohen-Kermani 2 classes': COHEN_KERMANI_2_CLASSES,
              'Cohen-Kermani 3 classes': COHEN_KERMANI_3_CLASSES}
COMPRESSION_FACTOR = 32
RANDOM_FOREST_PARAMS_GRID = {'n_estimators': [50, 100, 500, 1000]}
SVM_PARAMS_GRID = {'C': [0.5, 1, 10, 100], 'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
