import os 
import json
import random
import numpy as np

def class_append(class_arr, sample, dataset_dir):
    id = sample['id']
    status = sample['status']
    id_frames = []
    for frame in sample['frames']:
        frame_name = os.path.join(dataset_dir,id,frame['frame_name'])
        id_frames.append([frame_name,status])
    class_arr.append(id_frames)

def select(class_arr, split_ratio):
    length = len(class_arr)
    index_list = list(range(length))
    random.shuffle(index_list)
    split_index = int(split_ratio*length)
    train_list = [class_arr[i] for i in index_list[:split_index]]
    test_list = [class_arr[i] for i in index_list[split_index:]]
    return train_list, test_list

def split(dataset_dir, json_file='amap_traffic_annotations_train.json',split_ratio=0.8):
    json_dir = os.path.join(dataset_dir, json_file)
    file = open(json_dir,encoding='utf-8')
    file = json.loads(file.read())
    class_0 = []
    class_1 = []
    class_2 = []
    
    for sample in file['annotations']:
        status = sample['status']
        if status==0:
            class_append(class_0, sample, dataset_dir)
        elif status==1:
            class_append(class_1, sample, dataset_dir)
        else:
            class_append(class_2, sample, dataset_dir)
    class_0_train, class_0_test = select(class_0, split_ratio)
    class_1_train, class_1_test = select(class_1, split_ratio)
    class_2_train, class_2_test = select(class_2, split_ratio)
    train_list = np.concatenate([class_0_train,class_1_train,class_2_train])
    test_list = np.concatenate([class_0_test,class_1_test,class_2_test])
    return train_list, test_list

def write_txt(txt_dir, data_arr):
    file = open(txt_dir,'w')
    for id in data_arr:
        for sample in id:
            line = sample[0] + ' ' + str(sample[1]) + '\n'
            file.write(line)

if __name__ == "__main__":
    dataset_dir = '/home/apollo/ali-tianchi/dataset/train_dataset'
    train_list, test_list = split(dataset_dir)
    write_txt('./train_list.txt', train_list)
    write_txt('./test_list.txt', test_list)

