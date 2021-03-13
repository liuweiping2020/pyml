from dater.dateset import DataSet
import numpy as np


class DynMemNetDataSet(DataSet):
    def __init__(self):
        pass

    def load_train(self):
        pass

    def load_dev(self):
        pass

    def load_train_dev(self):
        pass

    def load_test(self):
        pass

    def set_predict_params(self):
        self.batch_size = 8
        self.story_length = 3
        self.sequence_length = 10
        pass

    def load_predict(self):
        pass

    def get_train_batch(self):
        pass

    def get_dev(self):
        pass

    def get_test(self):
        pass

    def get_predict(self):
        self.story = np.random.randn(self.batch_size, self.story_length, self.sequence_length)
        self.story[self.story > 0] = 1
        self.story[self.story <= 0] = 0
        self.query = np.random.randn(self.batch_size, self.sequence_length)
        self.query[self.query > 0] = 1
        self.query[self.query <= 0] = 0
        self.answer_single = np.sum(self.query, axis=1) + np.round(0.1 * np.sum(np.sum(self.story, axis=1), axis=1))
        pass
