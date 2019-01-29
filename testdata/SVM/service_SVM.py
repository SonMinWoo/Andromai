import os
import csv
import numpy as np
from tkinter import *
from tkinter import filedialog
from dictionary_api import *
from class_parse import *
from androguard.core.bytecodes.apk import APK
from sklearn.svm import SVC


#====================import maluse_api_dictionary
api_name = []
result = {}

for key,value in maluse_api.items():
    for element in value :
        api_name.append(key+" "+element)
sorted_api_list = sorted(api_name)

#====================import apk file
root = Tk()

root.filename =  filedialog.askopenfilename(initialdir = "E:/Images",title = "choose your file",filetypes = (("apk files","*.apk"),("all files","*.*")))

try:
    os.system("unzip -d /apktmp " +  root.filename)
    f = open("/apktmp/classes.dex",'rb')

except IOError as e:
    result[root.filename] = [-1 for i in range(77)]

#====================parse api data 
mm = f.read()
f.close()
hdr = header(mm)

string_ids = string_id_list(mm, hdr)
type_ids = type_id_list(mm, hdr)
method_ids = method_id_list(mm, hdr)

prt_api = {}
apilist = [0 for i in range(77)]

for i in range(len(method_ids)):
    (class_idx, proto_idx, name_idx) = method_ids[i]
    class_str = string_ids[type_ids[class_idx]]
    name_str = string_ids[name_idx]
    prt_api[class_str[1:]] = name_str.lower()
    for i in maluse_api:
        if class_str[1:].lower().find(i.encode('utf-8')) != -1:
            if 'NONE' in maluse_api[i]:
                cla_met = i + " " + 'NONE'
                apilist[sorted_api_list.index(cla_met)]=1
            if name_str.lower().decode('utf-8', errors = "ignore") in maluse_api[i]:
                cla_met = i + " " + name_str.lower().decode('utf-8')
                apilist[sorted_api_list.index(cla_met)]=1
result = apilist
#===================parse permission data

def get_perm(path):
    apkf = APK(path)
    apk_permissions = []
    for i in apkf.get_permissions():
        apk_permissions.append(i.split(".")[-1])
    return apk_permissions

permlist = get_perm(root.filename)

#===================calculate api_score

def csvopen(name):
    with open(name) as f:
        csvdata = csv.reader(f)
        data = [row for row in csvdata]
    f.close()
    return data


apiscore = csvopen('../../traindata/score/apiscore.csv')

def apk_apiscore_(apilist_,apiscore_):
    count = 0
    score = 0
    for i in range(len(apilist_)):
        if apilist_[i] == 1:
            count += 1
            score += float(apiscore_[i][1])
    if count == 0 : return 0.01
    else : return (score/count)

apk_apiscore = apk_apiscore_(apilist,apiscore)
#===================calculate perm_score

permscore = csvopen('../../traindata/score/permscore.csv')

def apk_permscore_(permlist_,permscore_):
    count = 0
    score = 0
    for element in permlist_:
        for row in permscore_:
            if row[0] == element:
                count += 1
                score += float(row[1])
            else : pass
    if count == 0 : return 0.01
    else : return (score/count)

apk_permscore = apk_permscore_(permlist,permscore)

print("permscore: ",apk_permscore)
print("apiscore : ",apk_apiscore)
#===================train data import

apkscore_perm_train = csvopen('../../traindata/score/apkscore_perm_train.csv')
apkscore_api_train = csvopen('../../traindata/score/apkscore_api_train.csv')

apkscore_perm_train_dict = {}
for row in apkscore_perm_train:
    apkscore_perm_train_dict[row[0]] = row[1] 

apkscoredict_train = {}
for row in apkscore_api_train:
    try : apkscoredict_train[row[0]] = [float(apkscore_perm_train_dict[row[0]]),float(row[1]),int(row[2])]
    except KeyError : pass



#svm
x_train =[]
y_train =[]
for key,value in apkscoredict_train.items():
    #value[0] : permission    value[1] : api    value[2] : label
    x_train.append([value[0],value[1]])
    y_train.append(value[2])
X_train = np.array(x_train)
Y_train = np.array(y_train)
print("===X===\n",X_train)
print("===Y===\n",Y_train)

X_test = np.array([apk_permscore,apk_apiscore])

clf = SVC(kernel='linear',gamma ='auto',C=0.1)
clf.fit(X_train,Y_train)

print("=================result==============")
print("classifiaction probablity: ", clf.score(X_train,Y_train))
predict = clf.predict([[apk_permscore,apk_apiscore]]) 
if predict == 0: print("normal apk file")
else : print("malicious apk file")
