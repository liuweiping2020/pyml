from modeler.resnetmodel import ResNetModel
from trainer.tftrainer import TFTrainer


class ResNetTrainer(TFTrainer):
    def __init__(self):
        self.batch_size = 32
        self.height, self.width = 224, 224
        self.num_batches= 100
        pass


    def get_data(self):

        pass


    def get_model(self):
        resNetModel=ResNetModel()
        self.net=resNetModel.net

        pass
    def train(self):
        for i in range(self.num_batches):
            self.session.run(self.net)

        pass




