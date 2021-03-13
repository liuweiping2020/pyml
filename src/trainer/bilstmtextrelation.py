from dater.bilstmtextrelationdataset import BiLstmTextRelationDataSet
from modeler.bilstmtextrelation import BiLstmTextRelationModel
from trainer.tftrainer import TFTrainer


class BiLstmTextRelationTrainer(TFTrainer):
    def __init__(self):
        self.num_classes = 2
        self.learning_rate = 0.001
        self.batch_size = 1
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 3
        self.vocab_size = 10000
        self.embed_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5
        super(BiLstmTextRelationTrainer, self).__init__()
        pass

    def get_model(self):
        self.textRNN = BiLstmTextRelationModel(self.num_classes, self.learning_rate, self.batch_size, self.decay_steps,
                                               self.decay_rate, self.sequence_length, self.vocab_size, self.embed_size,
                                               self.is_training)
        pass

    def get_data(self):
        biLstmTextRelationDataSet = BiLstmTextRelationDataSet()
        biLstmTextRelationDataSet.run_train()
        self.input_x = biLstmTextRelationDataSet.input_x
        self.input_y = biLstmTextRelationDataSet.input_y
        self.x1 = biLstmTextRelationDataSet.x1
        self.x2 = biLstmTextRelationDataSet.x2
        pass

    def get_feed_dict(self):
        self.feed_data = {self.textRNN.input_x: self.input_x, self.textRNN.input_y: self.input_y,
                          self.textRNN.dropout_keep_prob: self.dropout_keep_prob}
        pass

    def get_fetch_list(self):
        self.fetch_list = [self.textRNN.loss_val, self.textRNN.accuracy, self.textRNN.predictions,
                           self.textRNN.train_op]
        pass

    def train(self):
        for i in range(1000):
            loss, acc, predict, _ = self.session.run(self.fetch_list, feed_dict=self.feed_data)
            if i % 100 == 0:
                print(i, "x1:", self.x1, ";x2:", self.x2, ";loss:", loss, ";acc:", acc, ";label:", self.input_y,
                      ";prediction:", predict)

    pass


if __name__ == '__main__':
    blrt = BiLstmTextRelationTrainer()
    blrt.run()

    pass
