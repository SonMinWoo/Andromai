import csv
import random
import pandas as pd
from collections import OrderedDict

def dfopen(name):
#f means feature
    f_data = pd.read_csv(name)
    f_dict_ = f_data.to_dict('index')
    f_dict={}
    rmkeys = []
    
    for key,value in f_dict_.items():
        try :
            if value[random.choice(list(value))] == -1 : rmkeys.append(key)
        except KeyError : pass
    for key in rmkeys: del f_dict_[key]
    for key,value in f_dict_.items():
        f_dict[value["Unnamed: 0"]]=OrderedDict(sorted(value.items()))
        labelpos = tuple(f_dict[value["Unnamed: 0"]]).index('label')
        apknamepos = tuple(f_dict[value["Unnamed: 0"]]).index('Unnamed: 0')
        f_dict[value["Unnamed: 0"]] = [list(f_dict[value["Unnamed: 0"]].values()),f_dict[value["Unnamed: 0"]]['label']]
        del f_dict[value["Unnamed: 0"]][0][labelpos]
        del f_dict[value["Unnamed: 0"]][0][apknamepos]
    del f_dict_[0]['label']
    del f_dict_[0]['Unnamed: 0']
    f_list=sorted(list(f_dict_[0].keys()))
    return f_dict,f_list
    

apidict_train,apilist_train = dfopen('../api_parse/api_train.csv')
permdict_train,permlist_train = dfopen('../permission_parse/perm_train.csv')

def get_score(feature_dict,feature_list):
    count_nor = [0 for i in range(len(feature_list))]
    count_mal = [0 for i in range(len(feature_list))]
    score = [0 for i in range(len(feature_list))]

    for key,value in feature_dict.items():
        if value[1] == 0 :
            for i in range(len(value[0])) :
                if value[0][i] == 1: count_nor[i] += 1
                else : pass 
        else :
            for i in range(len(value[0])) :
                if value[0][i] == 1:  count_mal[i] += 1
                else : pass

    for i in range(len(count_nor)):
        if count_nor[i] == 0 : score[i] = 1
        else : score[i] = count_mal[i] / count_nor[i]

    featuredict_score = {}
    for key,value in feature_dict.items():
        count = 0
        score_ = 0
        for i in range(len(value[0])):
            if value[0][i] == 1 :
                score_ += score[i]
                count += 1
            else : pass
        if count == 0: featuredict_score[key] = [0.01,value[1]]
        else : featuredict_score[key] = [score_/count,value[1]]
    print(featuredict_score)
    return featuredict_score,score

apiscoredict,apiscore = get_score(apidict_train, apilist_train)
permscoredict,permscore = get_score(permdict_train, permlist_train)

f = open('apiscore.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f)
for i in range(len(apilist_train)):
    wr.writerow([apilist_train[i],apiscore[i]])
f.close()

f = open('permscore.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f)
for i in range(len(permlist_train)):
    wr.writerow([permlist_train[i],permscore[i]])
f.close()

f2 = open('apkscore_api_train.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f2)
for key,value in apiscoredict.items():
    wr.writerow([key,value[0],value[1]])
f2.close()

f2 = open('apkscore_perm_train.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f2)
for key,value in permscoredict.items():
    wr.writerow([key,value[0],value[1]])
f2.close()
