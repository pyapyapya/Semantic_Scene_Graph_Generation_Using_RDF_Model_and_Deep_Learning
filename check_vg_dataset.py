import json
from typing import List, Dict

from VRD import config


class CheckVGDataset:
    def __init__(self):
        """
        objects_list: List have 100 class
        predicates_list: List have 70 class
        """

        self.scene_graph_json = self.load_json_file(config.path['scene_graph_path'])
        # self.read_json_format(self.scene_graph_json)
        self.sample_data(self.scene_graph_json)

    @classmethod
    def sample_data(cls, json_file):
        for image_id, label in json_file.items():
            print(image_id)
            for spo in label:
                print('object_id', spo['objects']['object_id'])
                print('object_x', spo['objects']['x'])
                print('object_y', spo['objects']['y'])
                print('object_w', spo['objects']['w'])
                print('object_h', spo['objects']['h'])
                print('object_name', spo['objects']['name'])

                print('relationship_id', spo['relationships']['relationship_id'])
                print('relationship_x', spo['relationships']['x'])
                print('relationship_y', spo['relationships']['y'])
                print('relationship_w', spo['relationships']['w'])
                print('relationship_h', spo['relationships']['h'])
                print('relationship_name', spo['relationships']['name'])

                print('_id', spo['subjects']['subject_id'])
                print('subject_x', spo['subjects']['x'])
                print('subject_y', spo['subjects']['y'])
                print('subject_w', spo['subjects']['w'])
                print('subject_h', spo['subjects']['h'])
                print('subject_name', spo['subjects']['name'])
                break
            break

    @classmethod
    def check_dataset(cls, json_file):
        image_id_list: List = []
        subject_object: Dict = {}
        for image_id, label in json_file.items():
            image_id_list.append(image_id)
            for spo in label:
                subject_id = spo['subject']['category']
                object_id = spo['object']['category']
                predicate = spo['predicate']
                subject_object[subject_id] = object_id
        return image_id_list, subject_object

    @classmethod
    def read_json_format(cls, json_file: json):
        formatting_json = json.dumps(json_file, sort_keys=True, indent=4)
        print(formatting_json)

    @classmethod
    def load_json_file(cls, path):
        with open(path, 'r') as file:
            json_file = json.load(file)
        return json_file


class CheckTrainSet(CheckVGDataset):
    def __init__(self):
        super().__init__()
        self.json_train_data: json = self.load_json_file(config.path['json_train_dataset_path'])
        self.read_json_format(self.json_train_data)
        # self.sample_data(json_file=self.json_train_data)
        # image_id_list, subject_object = self.check_dataset(self.json_train_data)
        # print(subject_object)


class CheckTestSet(CheckVGDataset):
    def __init__(self):
        super().__init__()
        self.json_test_data: json = self.load_json_file(config.path['json_test_dataset_path'])
        # self.sample_data(json_file=self.json_train_data)


vg_dataset = CheckVGDataset()
# check_train_set = CheckTrainSet()
# check_test_set = CheckTestSet()
