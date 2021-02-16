import os

main_path = 'C:\dataset\\VRD\\json_dataset'
path = {
    # VRD DataSet
    'objects_path': os.path.join(main_path, 'objects.json'),
    'predicate_path': os.path.join(main_path, 'predicates.json'),
    'json_train_dataset_path': os.path.join(main_path, 'annotations_train.json'),
    'json_test_dataset_path': os.path.join(main_path, 'annotations_test.json'),
    # 'json_train_dataset_path': os.path.join('C:\dataset\\vrd\json_dataset\sg_dataset', 'sg_train_annotations.json'),
    # 'json_test_dataset_path': os.path.join('C:\dataset\\vrd\json_dataset\sg_dataset', 'sg_test_annotations.json'),
    'train_image_path': os.path.join('C:\dataset\\vrd\json_dataset\sg_dataset', 'sg_train_images'),
    'val_image_path': os.path.join('C:\dataset\\vrd\json_dataset\sg_dataset', 'sg_test_images'),

    # Make User DataSet
    'spo_train_path': os.path.join(main_path, 'spo_train.txt'),
    'spo_test_path': os.path.join(main_path, 'spo_test.txt'),
    'tag_representation_path': os.path.join(main_path, 'tag_representation.txt'),
    'train_two_tag_path': os.path.join(main_path, 'train_two_tag.pkl'),
    'test_two_tag_path': os.path.join(main_path, 'test_two_tag.pkl'),

    # Load DataSet
    'numberbatch_path': 'E:\ADD\ADD\\numberbatch-en.txt'
}
