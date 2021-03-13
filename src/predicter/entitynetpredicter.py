import numpy as np

from predicter.tfpredicter import TFPredicter


class EntityNetPredicter(TFPredicter):
    def __init__(self):
        self.num_classes = 15
        self.learning_rate = 0.001
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.sequence_length = 10
        self.vocab_size = 10000
        self.embed_size = 100
        self.hidden_size = 100
        self.is_training = False
        self.story_length = 3
        self.dropout_keep_prob = 1
        super(EntityNetPredicter, self).__init__()
        pass

    def set_ckpt_dir(self):
        self.ckpt_dir = '../../model/entitynet/'
        pass

    def get_fetch_list(self):
        predictions=self.session.graph.get_operation_by_name("predictions").outputs[0]
        self.feed_list = [predictions]
        pass

    def get_feed_dict(self):
        query=self.session.graph.get_operation_by_name("question").outputs[0]
        story=self.session.graph.get_operation_by_name("story").outputs[0]
        dropout_keep_prob=self.session.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.feed_back = {query: self.query, story: self.story,
                          dropout_keep_prob: self.dropout_keep_prob}

        pass


    def get_data(self):
        self.story = np.random.randn(self.batch_size, self.story_length, self.sequence_length)
        self.story[self.story > 0] = 1
        self.story[self.story <= 0] = 0
        self.query = np.random.randn(self.batch_size, self.sequence_length)  # [batch_size, sequence_length]
        self.query[self.query > 0] = 1
        self.query[self.query <= 0] = 0
        self.answer_single = np.sum(self.query, axis=1) + np.round(0.1 * np.sum(np.sum(self.story, axis=1), axis=1))
        # [batch_size].e.g. np.array([1, 0, 1, 1, 1, 2, 1, 1])
        pass

    def predict(self):
        for i in range(1500):
            predict = self.session.run(self.feed_list, feed_dict=self.feed_back)
            print(i, "query:", self.query, "=====================>")
            print(i, "label:", self.answer_single, "prediction:", predict)


if __name__ == '__main__':
    entityNetPredicter = EntityNetPredicter()
    entityNetPredicter.run()
    pass
