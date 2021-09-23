import os
import pickle
from typing import List, Dict

from check_vrd_dataset import CheckTrainSet, CheckTestSet, delete_overlap_train_set
import config


class TwoTag:
    def __init__(self, two_tag: Dict):
        self.two_tag: Dict = two_tag
        self.cnt = 0

    def __len__(self) -> int:
        return len(self.two_tag)

    def __call__(self, tag1, tag2):
        if tag1 not in self.two_tag:
            return None
        elif tag2 not in self.two_tag[tag1]:
            return None
        else:
            return self.two_tag[tag1][tag2]


class Preprocess:
    def __init__(self):
        self.main_path = config.main_path

        train_set = CheckTrainSet()
        test_set = CheckTestSet()

        train_set.make_dict_dataset(train_set.json_train_data)
        test_set.make_dict_dataset(test_set.json_test_data)

        train_spo_dict: Dict = train_set.get_spo_dict
        test_spo_dict: Dict = test_set.get_spo_dict
        # print(train_spo_dict)
        # print(test_spo_dict)

        self.objects_list: List = train_set.objects_list
        self.train_spo_dict, self.test_spo_dict = delete_overlap_train_set(train_set=train_spo_dict, test_set=test_spo_dict)

    def get_tag_representation(self, file_name: str):
        save_file_name = os.path.join(self.main_path, file_name)
        with open(config.path['numberbatch_path'], 'rt', encoding='UTF-8-sig') as numberbatch_file:
            with open(save_file_name, 'wt', encoding='UTF-8') as tag_representation_file:
                for line in numberbatch_file.readlines()[1:]:
                    object_name: str = line.split()[0]
                    if object_name in self.objects_list:
                        tag_representation_file.write(line)

    def make_spo(self, spo_dict: Dict, file_name: str):
        save_file_name = os.path.join(self.main_path, file_name)
        tag_dict: Dict = {}
        tag2tag_cnt = 0
        with open(save_file_name, 'wt', encoding='UTF-8') as spo_file:
            for subjects, object_predicate in spo_dict.items():
                for objects, predicate in object_predicate.items():
                    if subjects not in tag_dict:
                        tag_dict[subjects] = {}
                        tag_dict[subjects][objects] = tag2tag_cnt
                        tag2tag_cnt += 1

                    elif objects not in tag_dict[subjects]:
                        tag_dict[subjects][objects] = tag2tag_cnt
                        tag2tag_cnt += 1

                    if predicate.nonzero()[0].shape[0] == 0:
                        continue
                    else:
                        predicate = predicate.tolist()
                        spo_file.write(subjects + ' ' + objects + ' ' + ' '.join(map(str, predicate)) + '\n')
        return tag_dict

    def make_pkl(self, two_tag_dict: Dict, file_name: str):
        two_tag = TwoTag(two_tag_dict)
        save_file_name = os.path.join(self.main_path, file_name)
        with open(save_file_name, 'wb') as pkl:
            pickle.dump(two_tag, pkl)

    def file_process(self):
        self.get_tag_representation('tag_representation.txt')
        train_two_tag = self.make_spo(self.train_spo_dict, 'spo_train.txt')
        test_two_tag = self.make_spo(self.test_spo_dict, 'spo_test.txt')
        self.make_pkl(train_two_tag, 'train_two_tag.pkl')
        self.make_pkl(test_two_tag, 'test_two_tag.pkl')


def main():
    preprocess = Preprocess()
    preprocess.file_process()


# main()
