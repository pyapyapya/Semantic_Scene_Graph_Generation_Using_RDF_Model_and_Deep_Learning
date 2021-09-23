import json
from typing import List, Dict

import numpy as np

import config


class CheckDataset:
    def __init__(self):
        """
        objects_list: List have 100 class
        predicates_list: List have 70 class
        """

        self.objects_list: List = self.load_json_file(config.path['objects_path'])
        self.predicate_list: List = self.load_json_file(config.path['predicate_path'])

    def make_dict_dataset(self, json_file):
        """
        :param json_file:
        :return: image_id_list: List, spo_dict: Dict

        Make SPO data but, struct Subject-Object-Predicate

        train_relationship: 30355
        test_relationship: 7638
        """
        image_id_list: List = []
        spo_dict: Dict = {}

        n_dataset = 0
        total_cnt = 0
        subject_cnt = 0
        object_cnt = 0
        predicate_cnt = 0
        overlap_cnt = 0

        for image_id, label in json_file.items():
            image_id_list.append(image_id)
            total_cnt += len(label)
            for spo in label:
                subject_id: int = spo['subject']['category']
                object_id: int = spo['object']['category']
                predicate_id: int = spo['predicate']

                subject_name: str = self.objects_list[subject_id]
                object_name: str = self.objects_list[object_id]
                predicate_name: str = self.predicate_list[predicate_id]

                if subject_name not in spo_dict:
                    spo_dict[subject_name] = {}
                    spo_dict[subject_name][object_name]: np.array = np.zeros(len(self.predicate_list)).astype(int)
                    spo_dict[subject_name][object_name][predicate_id] = 1
                    subject_cnt += 1

                elif object_name not in spo_dict[subject_name]:
                    spo_dict[subject_name][object_name]: np.array = np.zeros(len(self.predicate_list)).astype(int)
                    spo_dict[subject_name][object_name][predicate_id] = 1
                    object_cnt += 1

                    # print(spo_dict[subject_name][object_name][predicate_id])
                elif spo_dict[subject_name][object_name][predicate_id] == 0:
                    spo_dict[subject_name][object_name][predicate_id] = 1
                    predicate_cnt += 1
                else:
                    overlap_cnt += 1

        print('object_cnt', object_cnt)
        total_cnt = subject_cnt+object_cnt+predicate_cnt
        print('total_cnt', total_cnt)
        return image_id_list, spo_dict

    @classmethod
    def sample_data(cls, json_file):
        for image_id, label in json_file.items():
            print(image_id)
            for spo in label:
                print('subject_id', spo['subject']['category'])
                print('subject_bbox', spo['subject']['bbox'])
                print('object_id', spo['object']['category'])
                print('object_bbox', spo['object']['bbox'])
                print('predicate', spo['predicate'])
                break
            break

    @classmethod
    def read_json_format(cls, json_file: json):
        formatting_json = json.dumps(json_file, sort_keys=True, indent=4)
        print(formatting_json)

    @classmethod
    def load_json_file(cls, path):
        with open(path, 'r') as file:
            json_file = json.load(file)
        return json_file


class CheckTrainSet(CheckDataset):
    def __init__(self):
        super().__init__()
        self.json_train_data: json = self.load_json_file(config.path['json_train_dataset_path'])
        # self.read_json_format(self.json_train_data)
        self.sample_data(json_file=self.json_train_data)
        self.image_id_list, self.spo_dict = self.make_dict_dataset(self.json_train_data)

    @property
    def get_image_id_list(self) -> List:
        return self.image_id_list

    @property
    def get_spo_dict(self) -> Dict:
        return self.spo_dict


class CheckTestSet(CheckDataset):
    def __init__(self):
        super().__init__()
        self.json_test_data: json = self.load_json_file(config.path['json_test_dataset_path'])
        self.image_id_list, self.spo_dict = self.make_dict_dataset(self.json_test_data)
        # self.sample_data(json_file=self.json_test_data)

    @property
    def get_spo_dict(self):
        return self.spo_dict


def delete_overlap_train_set(train_set: Dict, test_set: Dict):
    only_test_spo_cnt = 0
    delete_cnt = 0
    for test_subject, test_objects_predicate in test_set.items():
        if test_subject not in train_set:
            only_test_spo_cnt += cnt[0].shape[0]
            continue
        for test_objects, test_predicate in test_objects_predicate.items():
            if test_objects not in train_set[test_subject]:
                cnt = test_set[test_subject][test_objects].nonzero()
                only_test_spo_cnt += cnt[0].shape[0]
                continue
            test_idx = test_set[test_subject][test_objects].nonzero()[0]
            for idx in test_idx:
                if test_set[test_subject][test_objects][idx] == 1:
                    train_set[test_subject][test_objects][idx] = 1
                    test_set[test_subject][test_objects][idx] = 0
                    delete_cnt += 1
                else:
                    only_test_spo_cnt += 1

            # for idx in test_idx:
            #     if train_set[test_subject][test_objects][idx] == 1:
            #         train_set[test_subject][test_objects][idx] = 0
            #         delete_cnt += 1
            #     else:
            #         only_test_spo_cnt += 1

    total_test_data = only_test_spo_cnt + delete_cnt
    print('only_test_spo_cnt', only_test_spo_cnt)
    print('delete_cnt', delete_cnt)
    print('total_test_data', total_test_data)

    train_idx = 0
    train_tag_idx = 0

    train_dict: Dict = {}
    for s, po in train_set.items():
        for p, o in po.items():
            for o_idx, o_value in enumerate(o):
                if train_set[s][p][o_idx] == 1:
                    train_idx += 1

    test_idx = 0
    for s, po in test_set.items():
        for p, o in po.items():
            for o_idx, o_value in enumerate(o):
                if test_set[s][p][o_idx] == 1:
                    test_idx += 1

    print('train_idx', train_idx)
    print('test_idx', test_idx)

    return train_set, test_set


def main():
    check_train_set = CheckTrainSet()
    check_test_set = CheckTestSet()

    train_spo_dict: Dict = check_train_set.get_spo_dict
    test_spo_dict: Dict = check_test_set.get_spo_dict
    delete_overlap_train_set(train_spo_dict, test_spo_dict)


# main()
