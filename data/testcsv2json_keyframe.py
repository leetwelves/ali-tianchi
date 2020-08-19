import csv
from pandas import Series,DataFrame
import json
import os

testjson_dir = '/home/apollo/ali-tianchi/results'
path =  os.path.join(testjson_dir,'amap_traffic_annotations_test.json')
jfile = open(path,encoding='utf-8')
jfile = json.loads(jfile.read())
key_id_path = []
for sample in jfile['annotations']:
    id = sample['id']
    key_frame = sample['key_frame']
    key_id_path_temp = '/home/apollo/ali-tianchi/dataset/test_dataset/{}/{}'.format(id,key_frame)
    key_id_path.append(key_id_path_temp)    


tfile = csv.reader(open('/home/apollo/ali-tianchi/results/model_test_38.csv','r'))
results0=[]
results1=[]
for row in tfile:
    results0.extend([row[0]])
    results1.extend([row[1]])
dict1=dict(zip(results0,results1))    


result=[]
for ii in range(600):
    result_temp = dict1.get(key_id_path[ii])
    result.append(result_temp)


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

