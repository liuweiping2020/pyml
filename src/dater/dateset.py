class DataSet(object):
    def __init__(self):
        self.pre_data_set()
        pass

    def pre_data_set(self):
        pass

    def run_train(self):
        self.set_train_params()
        self.load_train()
        self.get_train_batch()
        pass

    def set_train_params(self):
        pass

    def load_train(self):
        pass

    def get_train_batch(self):
        pass

    def run_dev(self):
        self.set_dev_params()
        self.load_dev()
        self.get_dev()
        pass

    def set_dev_params(self):
        pass

    def load_dev(self):
        pass

    def get_dev(self):
        pass

    def run_test(self):
        self.set_test_params()
        self.load_test()
        self.get_test()
        pass

    def set_test_params(self):
        pass

    def load_test(self):
        pass

    def get_test(self):
        pass

    def run_predict(self):
        self.set_predict_params()
        self.load_predict()
        self.get_predict()
        pass

    def set_predict_params(self):
        pass

    def load_predict(self):
        pass

    def get_predict(self):
        pass
