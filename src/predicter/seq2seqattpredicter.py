import copy
import random

import numpy as np

from predicter.tfpredicter import TFPredicter


class Seq2SeqAttPredicter(TFPredicter):
    def __init__(self):
        self.num_classes = 9 + 2  # additional two classes:one is for _GO, another is for _END
        self.learning_rate = 0.0001
        self.batch_size = 1
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 5
        self.vocab_size = 300
        self.embed_size = 100  # 100
        self.hidden_size = 100
        self.is_training = True
        self.dropout_keep_prob = 1  # 0.5 #num_sentences
        self.decoder_sent_length = 6
        self.l2_lambda = 0.0001
        super(Seq2SeqAttPredicter,self).__init__()
        pass
    def get_data(self):
        label_list = self.get_unique_labels()
        self.input_x = np.array([label_list], dtype=np.int32)  # [2,3,4,5,6]
        self.label_list_original = copy.deepcopy(label_list)
        label_list.reverse()
        self.decoder_input = np.array([[0] + label_list], dtype=np.int32)  # [[0,2,3,4,5,6]]
        self.input_y_label = np.array([label_list + [1]], dtype=np.int32)  # [[2,3,4,5,6,1]]
        pass

    def get_unique_labels(self):
        x = [2, 3, 4, 5, 6]
        random.shuffle(x)
        return x

    def set_ckpt_dir(self):
        self.ckpt_dir = '../../model/seq2seqatt/'

        pass
    def get_fetch_list(self):
        predictions=self.session.graph.get_operation_by_name("predictions").outputs[0]
        self.fetch_list = [predictions]
        pass
    def get_feed_dict(self):
        input_x=self.session.graph.get_operation_by_name("input_x").outputs[0]
        decoder_input = self.session.graph.get_operation_by_name("decoder_input").outputs[0]
        input_y_label = self.session.graph.get_operation_by_name("input_y_label").outputs[0]
        dropout_keep_prob = self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.feed_dict = {input_x: self.input_x,
                          decoder_input: self.decoder_input,
                          input_y_label: self.input_y_label,
                          dropout_keep_prob: self.dropout_keep_prob}

        pass

    def predict(self):
        for i in range(150):
            predict = self.session.run(self.fetch_list, feed_dict=self.feed_dict)
            print(i,  "prediction:", predict)

        pass


if __name__ == '__main__':
    seq2SeqAttPredicter=Seq2SeqAttPredicter()
    seq2SeqAttPredicter.run()

    pass