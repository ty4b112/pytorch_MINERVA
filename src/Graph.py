import torch
import numpy as np
from collections import defaultdict
import copy


class Knowledge_graph():
    def __init__(self, option, data_loader, data):
        self.option = option
        self.data = data
        self.data_loader = data_loader
        self.out_array = None
        self.all_correct = None
        self.construct_graph()

    # 根据原始数据构建知识图谱，out_array存储每个节点向外的出口路径数组
    def construct_graph(self):
        all_out_dict = defaultdict(list)
        for head, relation, tail in self.data:
            all_out_dict[head].append((relation, tail))

        all_correct = defaultdict(set)
        out_array = np.ones((self.option.num_entity, self.option.max_out, 2), dtype=np.int64)
        out_array[:, :, 0] *= self.data_loader.relation2num["Pad"]
        out_array[:, :, 1] *= self.data_loader.entity2num["Pad"]
        more_out_count = 0
        for head in all_out_dict:
            out_array[head, 0, 0] = self.data_loader.relation2num["Equal"]
            out_array[head, 0, 1] = head
            num_out = 1
            for relation, tail in all_out_dict[head]:
                if num_out == self.option.max_out:
                    more_out_count += 1
                    break
                out_array[head, num_out, 0] = relation
                out_array[head, num_out, 1] = tail
                num_out += 1
                all_correct[(head, relation)].add(tail)
        self.out_array = torch.from_numpy(out_array)
        self.all_correct = all_correct
        print("more_out_count", more_out_count)
        if self.option.use_cuda:
            self.out_array = self.out_array.cuda()

    # 获取从图谱上current_entities的out_relations, out_entities
    def get_out(self, current_entities, start_entities, queries, answers, all_correct, step):
        ret = copy.deepcopy(self.out_array[current_entities, :, :])
        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]
                mask = queries[i].eq(relations) & answers[i].eq(entities)
                #mask = queries[i].eq(relations)
                ret[i, :, 0][mask] = self.data_loader.relation2num["Pad"]
                ret[i, :, 1][mask] = self.data_loader.entity2num["Pad"]

            if step == self.option.max_step_length - 1:
                relations = ret[i, :, 0]
                entities = ret[i, :, 1]
                answer = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct[i] and entities[j] != answer:
                        relations[j] = self.data_loader.relation2num["Pad"]
                        entities[j] = self.data_loader.entity2num["Pad"]

        return ret

    def get_next(self, current_entities, out_ids):
        next_out = self.out_array[current_entities, :, :]
        next_out_list = list()
        for i in range(out_ids.shape[0]):
            next_out_list.append(next_out[i, out_ids[i]])
        next_out = torch.stack(next_out_list)
        next_entities = next_out[:, 1]
        return next_entities

    def get_all_correct(self, start_entities_np, relations_np):
        all_correct = list()
        for i in range(start_entities_np.shape[0]):
            all_correct.append(self.all_correct[(start_entities_np[i], relations_np[i])])
        return all_correct