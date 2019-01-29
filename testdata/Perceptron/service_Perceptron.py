import os
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog
from dictionary_api import *
from class_parse import *
from collections import OrderedDict
from androguard.core.bytecodes.apk import APK
from sklearn.linear_model import Perceptron

#====================import maluse_api_dictionary
api_name = []
result = {}

maluse_api_ = OrderedDict(sorted(maluse_api.items()))
for key,value in maluse_api_.items():
    for element in value :
        api_name.append(key+" "+element)
sorted_api_list = sorted(api_name)



for i in maluse_api_:
    name = i
    apilist = [0 for i in range(77)]


#====================import test apk file
root = Tk()

root.filename =  filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("apk files","*.apk"),("all files","*.*")))

try:
    os.system("unzip -d /apktmp " +  root.filename)
    f = open("/apktmp/classes.dex",'rb')

except IOError as e:
    result[root.filename] = [-1 for i in range(77)]

#====================parse api testdata 
mm = f.read()
f.close()
hdr = header(mm)

string_ids = string_id_list(mm, hdr)
type_ids = type_id_list(mm, hdr)
method_ids = method_id_list(mm, hdr)

prt_api = {}

for i in range(len(method_ids)):
    (class_idx, proto_idx, name_idx) = method_ids[i]
    class_str = string_ids[type_ids[class_idx]]
    name_str = string_ids[name_idx]
    prt_api[class_str[1:]] = name_str.lower()
    for i in maluse_api_:
        if class_str[1:].lower().find(i.encode('utf-8')) != -1:
            if 'NONE' in maluse_api_[i]:
                cla_met = i + " " + 'NONE' #cla_met: class+method
                apilist[sorted_api_list.index(cla_met)]=1
            if name_str.lower().decode('utf-8', errors = "ignore") in maluse_api_[i]:
                cla_met = i + " " + name_str.lower().decode('utf-8')
                apilist[sorted_api_list.index(cla_met)]=1
result = apilist

apilist_byname = []
for i in range(len(sorted_api_list)):
    if apilist[i] == 1 : apilist_byname.append(sorted_api_list[i])


#===================parse permission testdata

def get_perm(path):
    apkf = APK(path)
    apk_permissions = []
    for i in apkf.get_permissions():
        apk_permissions.append(i.split(".")[-1])
    return apk_permissions

permlist_byname = get_perm(root.filename)


#===================traindata preprocessing

def dfopen(name):
    df = pd.read_csv(name,index_col = 'Unnamed: 0') 
    df = df.sort_index(axis=1)
    return df

perm_train = dfopen('../../traindata/permission_parse/perm_train.csv')
api_train = dfopen('../../traindata/api_parse/api_train.csv')
del api_train['label']
result = pd.merge(perm_train, api_train, left_index = True, right_index=True ,how='inner')
result = result.sort_index(axis=1)
merged_train = result.to_dict('index')
mergedict={}
for key,value in merged_train.items():
    mergedict[key]=OrderedDict(sorted(value.items()))
    label = mergedict[key]['label']
    del mergedict[key]['label']
    names = list(mergedict[key].keys())
    mergedict[key] = [list(mergedict[key].values()) , label]


x_list = []
y_list = []
for key,value in mergedict.items():
    x_list.append(value[0])
    y_list.append(value[1])
train_x = np.array(x_list)
train_y = np.array(y_list)

#===================testdata preprocessing

X_test_tmp = [0 for i in range(len(names))]
for element in permlist_byname:
    try : X_test_tmp[names.index(element)] = 1
    except KeyError : print(element)
for element in apilist_byname:
    try : X_test_tmp[names.index(element)] = 1
    except KeyError : print(element)

test_x = np.array([X_test_tmp])
print(test_x)
#==================perceptron

clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(train_x,train_y)

print("=================result==============")
print("classifiaction probablity: ", clf.score(train_x,train_y))
predict = clf.predict(test_x)
if predict == 0: print("normal apk file")
else : print("malicious apk file")
