from functools import total_ordering
import pandas as pd


@total_ordering
class Result:
    class_set = None

    def __init__(self, auc, confusion_matrix, classification_report, accuracy=None):
        if accuracy is None:
            self.__set_accuracy(classification_report)
        else:
            self.accuracy = accuracy
        self.auc = auc
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report

    def accuracy_to_string(self):
        return '%.2f' % (self.accuracy * 100) + '%'

    def auc_to_string(self):
        return '%.3f' % self.auc

    def __set_accuracy(self, classification_report):
        accuracy = classification_report['accuracy']
        self.accuracy = accuracy

    def __confusion_matrix_to_df(self):
        class_set = Result.class_set
        confusion_matrix_df = pd.DataFrame(self.confusion_matrix, columns=class_set, index=class_set)
        return confusion_matrix_df

    def __classification_report_to_df(self):
        class_set = Result.class_set
        report_df = pd.DataFrame(self.classification_report).drop(['accuracy'], axis=1)
        class_index_name_dict = dict()
        for i, class_name in enumerate(class_set):
            class_index_name_dict['%.1f' % float(i)] = class_name
        report_df = report_df.rename(columns=class_index_name_dict)
        return report_df

    def __str__(self):
        output = f'Accuracy: {self.accuracy_to_string()}\tAUC: {self.auc_to_string()}\n'
        output += f'\nConfusion matrix:\n{self.__confusion_matrix_to_df()}\n'
        output += f'\nClassification report:\n{self.__classification_report_to_df()}'
        return output

    def __eq__(self, other):
        return self.accuracy == other.accuracy

    def __lt__(self, other):
        return self.accuracy < other.accuracy

    def __add__(self, other):
        if not isinstance(other, Result):
            return self
        self_attrs = vars(self)
        other_attrs = vars(other)
        attrs_sum = Result.__add_attrs(self_attrs, other_attrs)
        result_sum = Result(**attrs_sum)
        return result_sum

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError
        self_attrs = vars(self)
        attrs_division_result = Result.__divide_attrs(self_attrs, other)
        division_result = Result(**attrs_division_result)
        return division_result

    @staticmethod
    def __add_attrs(attr1, attr2):
        attr_sum = dict()
        for key, value in attr1.items():
            if isinstance(value, dict):
                attr_sum[key] = Result.__add_attrs(value, attr2[key])
            else:
                attr_sum[key] = value + attr2[key]
        return attr_sum

    @staticmethod
    def __divide_attrs(attr1, denominator):
        division_result = dict()
        for key, value in attr1.items():
            if isinstance(value, dict):
                division_result[key] = Result.__divide_attrs(value, denominator)
            else:
                division_result[key] = value / denominator
        return division_result
