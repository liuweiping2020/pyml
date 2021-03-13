class TFModel(object):
    def __init__(self):
        self.run()
        pass

    def build(self):
        pass

    def add_placeholder(self):
        pass

    def cal_loss(self):
        pass

    def recon_optimize(self):
        pass

    def cal_accuracy(self):
        pass

    def cal_predict(self):
        pass

    def run(self):
        self.add_placeholder()
        self.build()
        self.cal_loss()
        self.recon_optimize()
        self.cal_predict()
        self.cal_accuracy()
        pass

