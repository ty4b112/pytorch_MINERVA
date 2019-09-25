import torch
import numpy as np


class Environment():
    def __init__(self, option, graph, data, mode="train"):
        self.mode = mode
        self.data = data
        self.graph = graph
        self.option = option

        self.data_array = torch.from_numpy(data)
        if self.option.use_cuda:
            self.data_array = self.data_array.cuda()

    def get_next_batch(self):
        if self.mode == 'train':
            return self.yield_next_batch_train()
        else:
            return self.yield_next_batch_test()

    def yield_next_batch_train(self):
        while True:
            batch_idx = np.random.randint(0, len(self.data_array), size=self.option.batch_size)
            #batch_idx = np.arange(0, self.option.batch_size)
            batch = self.data_array[batch_idx, :]
            start_entities = batch[:, 0]
            relations = batch[:, 1]
            answers = batch[:, 2]
            if self.option.use_cuda:
                start_entities_np = start_entities.cpu().numpy()
                relations_np = relations.cpu().numpy()
            else:
                start_entities_np = start_entities.numpy()
                relations_np = relations.numpy()
            all_correct = self.graph.get_all_correct(start_entities_np, relations_np)
            start_entities, relations, answers, all_correct = self.data_times(start_entities, relations, answers,
                                                                              all_correct, "train")

            yield start_entities, relations, answers, all_correct

    def yield_next_batch_test(self):
        test_data_count = len(self.data)
        current_idx = 0
        while True:
            if current_idx == test_data_count:
                return
            if test_data_count - current_idx > self.option.batch_size:
                batch_idx = np.arange(current_idx, current_idx + self.option.batch_size)
                current_idx += self.option.batch_size
            else:
                batch_idx = np.arange(current_idx, test_data_count)
                current_idx = test_data_count

            batch = self.data_array[batch_idx, :]
            start_entities = batch[:,0]
            relations = batch[:, 1]
            answers = batch[:, 2]

            if self.option.use_cuda:
                start_entities_np = start_entities.cpu().numpy()
                relations_np = relations.cpu().numpy()
            else:
                start_entities_np = start_entities.numpy()
                relations_np = relations.numpy()

            all_correct = self.graph.get_all_correct(start_entities_np, relations_np)
            _start_entities, _relations, _answers, all_correct = self.data_times(start_entities, relations, answers,
                                                                                 all_correct, "test")

            yield _start_entities, _relations, _answers, start_entities, relations, answers, all_correct

    def data_times(self, start_entities, relations, answers, all_correct, mode):
        if mode == "train":
            times = self.option.train_times
        else:
            times = self.option.test_times

        start_entities = start_entities.repeat_interleave(times)
        relations = relations.repeat_interleave(times)
        answers = answers.repeat_interleave(times)
        new_all_correct = list()
        for item in all_correct:
            for _ in range(times):
                new_all_correct.append(item)
        return start_entities, relations, answers, new_all_correct