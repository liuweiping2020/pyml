from dater.dateset import DataSet
import random
import numpy as np


class BiLstmTextRelationDataSet(DataSet):
    def __init__(self):
        pass

    def get_train_batch(self):
        self.x1 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.x2 = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.input_x = np.array([[self.x1, 0, self.x2]])
        label = 0
        if np.abs(self.x1 - self.x2) < 3: label = 1
        self.input_y = np.array([label])
        pass