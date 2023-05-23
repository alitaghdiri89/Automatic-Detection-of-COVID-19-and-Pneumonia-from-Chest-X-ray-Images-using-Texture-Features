import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import constants as const
from db_creator import get_db_file_path
from result import Result
import math
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
from itertools import product
from sklearn.naive_bayes import GaussianNB


def load_database(class_count):
    db_file_path = get_db_file_path(class_count)
    df = pd.read_csv(db_file_path)
    return df


def create_train_test_dfs(data_frame):
    X = data_frame.drop(['class'], axis=1)
    y = data_frame['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=const.TRAIN_SIZE, stratify=y)
    return X_train, X_test, y_train, y_test


def get_svm_roc_auc(X_test, y_test, clf, class_count):
    y_scores = clf.decision_function(X_test)
    if class_count == 2:
        auc = roc_auc_score(y_test, y_scores[:, 1])
    else:
        ovr_auc_list = list()
        for i in range(class_count):
            i_vs_rest_y = y_test.apply(lambda label: int(label == i))
            curr_auc = roc_auc_score(i_vs_rest_y, y_scores[:, i])
            ovr_auc_list.append(curr_auc)
        auc = sum(ovr_auc_list) / len(ovr_auc_list)
    return auc


def get_roc_auc(X_test, y_test, clf, clf_type):
    class_count = len(np.unique(y_test))
    if clf_type == 'svm':
        return get_svm_roc_auc(X_test, y_test, clf, class_count)
    y_probabilities = clf.predict_proba(X_test)
    if class_count == 2:
        auc = roc_auc_score(y_test, y_probabilities[:, 1])
    else:
        auc = roc_auc_score(y_test, y_probabilities, multi_class='ovr', average='macro')
    return auc


def get_result(X_test, y_test, y_pred, clf, clf_type):
    curr_confusion_matrix = confusion_matrix(y_test, y_pred)
    curr_classification_report = classification_report(y_test, y_pred, output_dict=True)
    curr_auc = get_roc_auc(X_test, y_test, clf, clf_type)
    curr_result = Result(curr_auc, curr_confusion_matrix, curr_classification_report)
    return curr_result


def do_tree(df):
    results = list()
    total_time = 0
    print('DECISION TREE')
    for i in range(const.NUMBER_OF_EXPERIMENTS):
        print(f'Experiment No.: {i}')
        X_train, X_test, y_train, y_test = create_train_test_dfs(df)
        start_time = time.time()
        preliminary_tree = DecisionTreeClassifier(random_state=42)
        preliminary_tree.fit(X_train, y_train)
        ccp_path = preliminary_tree.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = ccp_path.ccp_alphas
        experiment_result = None
        for alpha in tqdm(ccp_alphas):
            tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            curr_result = get_result(X_test, y_test, y_pred, tree, 'tree')
            if experiment_result is None or curr_result > experiment_result:
                experiment_result = curr_result
        results.append(experiment_result)
        total_time += time.time() - start_time
    best_result = max(results)
    mean_result = sum(results) / const.NUMBER_OF_EXPERIMENTS
    mean_time = total_time / const.NUMBER_OF_EXPERIMENTS
    return dict, best_result, mean_result, mean_time


def classify_single_param_set(clf_class, param_set, df, clf_type):
    curr_results = list()
    total_time = 0
    print(param_set)
    for _ in tqdm(range(const.NUMBER_OF_EXPERIMENTS)):
        X_train, X_test, y_train, y_test = create_train_test_dfs(df)
        clf = clf_class(**param_set)
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        curr_result = get_result(X_test, y_test, y_pred, clf, clf_type)
        curr_results.append(curr_result)
        total_time += time.time() - start_time
    best_result = max(curr_results)
    mean_result = sum(curr_results) / const.NUMBER_OF_EXPERIMENTS
    print(mean_result.accuracy_to_string())
    mean_time = total_time / const.NUMBER_OF_EXPERIMENTS
    return param_set, best_result, mean_result, mean_time


def classify(clf_type, class_count):
    Result.class_set = const.CLASS_SETS[f'Cohen-Kermani {class_count} classes']
    df = load_database(class_count)
    results = list()
    if clf_type == 'tree':
        results.append(do_tree(df))
        return results
    if clf_type == 'random_forest':
        clf_class = RandomForestClassifier
        clf_params_grid = {'n_estimators': [50, 100, 500, 1000]}
    elif clf_type == 'svm':
        clf_class = SVC
        clf_params_grid = {'C': [0.5, 1, 10, 100], 'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    elif clf_type == 'knn':
        clf_class = KNeighborsClassifier
        data_count = df.shape[0]
        clf_params_grid = {'n_neighbors': list(range(2, math.ceil(math.sqrt(data_count)) + 1)), 'metric': ['euclidean']}
    elif clf_type == 'gaussian_naive_bayes':
        clf_class = GaussianNB
        clf_params_grid = dict()
    else:
        raise Exception("clf_type invalid!")
    param_names = clf_params_grid.keys()
    for param_values in product(*clf_params_grid.values()):
        clf_params = dict(zip(param_names, param_values))
        results.append(classify_single_param_set(clf_class, clf_params, df, clf_type))
    return results
