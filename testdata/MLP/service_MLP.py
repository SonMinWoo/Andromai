import os
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog
from dictionary_api import *
from class_parse import *
from collections import OrderedDict
from androguard.core.bytecodes.apk import APK
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
    if value[1] == 0 : y_list.append([1,0])
    else : y_list.append([0,1])

train_x_all = np.array(x_list)
train_y_all = np.array(y_list)

#===================testdata preprocessing

X_test_tmp = [0 for i in range(len(names))]
for element in permlist_byname:
    try : X_test_tmp[names.index(element)] = 1
    except KeyError : print(element)
for element in apilist_byname:
    try : X_test_tmp[names.index(element)] = 1
    except KeyError : print(element)

test_x = np.array([X_test_tmp])



#============================================ MLP

# TF graph
x = tf.placeholder(tf.float32,[None,len(train_x_all[0])])
y = tf.placeholder(tf.float32,[None,2])

W1 = tf.Variable(tf.random_uniform([len(train_x_all[0]), 500], -1., 1.))
W2 = tf.Variable(tf.random_uniform([500, 400], -1., 1.))
W3 = tf.Variable(tf.random_uniform([400, 300], -1., 1.))
W4 = tf.Variable(tf.random_uniform([300, 200], -1., 1.))
W5 = tf.Variable(tf.random_uniform([200, 100], -1., 1.))
W6 = tf.Variable(tf.random_uniform([100, 200], -1., 1.))
W7 = tf.Variable(tf.random_uniform([200, 300], -1., 1.))
W8 = tf.Variable(tf.random_uniform([300, 400], -1., 1.))
W9 = tf.Variable(tf.random_uniform([400, 500], -1., 1.))
W10 = tf.Variable(tf.random_uniform([500, 2], -1., 1.))

b1 = tf.Variable(tf.random_uniform([500], -1., 1.))
b2 = tf.Variable(tf.random_uniform([400], -1., 1.))
b3 = tf.Variable(tf.random_uniform([300], -1., 1.))
b4 = tf.Variable(tf.random_uniform([200], -1., 1.))
b5 = tf.Variable(tf.random_uniform([100], -1., 1.))
b6 = tf.Variable(tf.random_uniform([200], -1., 1.))
b7 = tf.Variable(tf.random_uniform([300], -1., 1.))
b8 = tf.Variable(tf.random_uniform([400], -1., 1.))
b9 = tf.Variable(tf.random_uniform([500], -1., 1.))
b10 = tf.Variable(tf.random_uniform([2], -1., 1.))

L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L8 = tf.nn.relu(tf.matmul(L7, W8) + b8)
L9 = tf.nn.relu(tf.matmul(L8, W9) + b9)

pred = tf.add(tf.matmul(L9, W10), b10)


reg = 0
reg += tf.nn.l2_loss(W1)
reg += tf.nn.l2_loss(W2)
reg += tf.nn.l2_loss(W3)
reg += tf.nn.l2_loss(W4)
reg += tf.nn.l2_loss(W5)
reg += tf.nn.l2_loss(W6)
reg += tf.nn.l2_loss(W7)
reg += tf.nn.l2_loss(W8)
reg += tf.nn.l2_loss(W9)
reg += tf.nn.l2_loss(W10)


learning_rate = 0.01
batch_size = 500

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(cost + 0.001 * reg)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()


def cross_validate(session, split_size=3):
  results = []
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_x_all, train_y_all):
    train_x = train_x_all[train_idx]
    train_y = train_y_all[train_idx]
    val_x = train_x_all[val_idx]
    val_y = train_y_all[val_idx]
    run_train(session, train_x, train_y)
    results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
  return results

def run_train(session, train_x, train_y):
  print("\nStart training")
  session.run(init)
  for epoch in range(100):
    total_batch = int(train_x.shape[0] / batch_size)
    for i in range(total_batch):
      batch_x = train_x[i*batch_size:(i+1)*batch_size]
      batch_y = train_y[i*batch_size:(i+1)*batch_size]
      _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
      if i % 50 == 0:
        print("Epoch #%d step=%d cost=%f" % (epoch, i, c))

with tf.Session() as session:
  result = cross_validate(session)
  print("classifiaction probablity(Cross-validation): %s" % result)
  prediction = tf.argmax(pred,1)
  predictval = session.run(prediction, feed_dict={x:test_x})



print("=================result==============")
if predictval == 0: print("normal apk file")
else : print("malicious apk file")
