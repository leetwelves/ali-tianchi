import csv
from pandas import Series,DataFrame
import json
import os
from collections import Counter

tfile = csv.reader(open('/home/apollo/ali-tianchi/results/model_test_38.csv','r'))
results0=[]
results1=[]
for row in tfile:
    results0.extend([row[0]])
    results1.extend([row[1]])
dict1=dict(zip(results0,results1))    

testjson_dir = '/home/apollo/ali-tianchi/results'
path =  os.path.join(testjson_dir,'amap_traffic_annotations_test.json')
jfile = open(path,encoding='utf-8')
jfile = json.loads(jfile.read())
all_id_value = []
for sample in jfile['annotations']:
    id = sample['id']
    id_value = []
    for frames in sample['frames']:
        frames_path = os.path.join('/home/apollo/ali-tianchi/dataset/test_dataset',id,frames['frame_name'])
        id_value_temp = dict1.get(frames_path)
        id_value.extend(id_value_temp)
    all_id_value.append(id_value)
Counter1=Counter(all_id_value[594])
print(Counter1)

result = []
for ii in range(600):
    counter_temp = Counter(all_id_value[ii])
    if counter_temp['2'] == 0 & counter_temp['1'] ==0 :
        result.append(0)
    elif counter_temp['2'] <= counter_temp['1']:
        result.append(1)
    else :
        result.append(2)
print (result)


json_path = "/home/apollo/ali-tianchi/results/amap_traffic_annotations_test.json"
out_path = "/home/apollo/ali-tianchi/results/amap_traffic_annotations_test_result.json"
# result 是你的结果, key是id, value是status
with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
    json_dict = json.load(f)
    data_arr = json_dict["annotations"]  
    new_data_arr = [] 
    for data in data_arr:
        id_ = int(data["id"])-1
        data["status"] = int(result[id_])
        new_data_arr.append(data)
    json_dict["annotations"] = new_data_arr
    json.dump(json_dict, w)






