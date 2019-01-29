from androguard.core.bytecodes.apk import APK
import os
import pandas as pd

#get dict of permissions in the path
def get_perm(path):
    filenames = os.listdir(path)
    path_permissions = {}
    for filename in filenames:
        filepath = path + "/" + filename
        apkf = APK(filepath)
        apk_permissions = []
        for i in apkf.get_permissions():
            apk_permissions.append(i.split(".")[-1])
        path_permissions[filename]=apk_permissions
        print(path_permissions) 
    return path_permissions

current_path = os.getcwd()

normal_path = current_path + "/../" + "normal_apk"
mal_path = current_path + "/../" + "malicious_apk"

permdict_normal = get_perm(normal_path)
permdict_mal = get_perm(mal_path)

#get permissions list 
def permissions_list(permdict_normal_,permdict_mal_):
    permissions = []
    for key,value in permdict_normal_.items():
        for element in value:
            permissions.append(element)
    permissions = list(set(permissions))

    for key,value in permdict_mal_.items():
        for element in value:
            permissions.append(element)
    permissions = list(set(permissions))
    permissions = sorted(permissions)
    return permissions

permissions = permissions_list(permdict_normal,permdict_mal)
print(permissions)


def permdict_to_01(permdict,perm,label):
    dict01={}
    for key,value in permdict.items():
        temp_list=[0 for i in range(len(perm))]
        for element in value:
            temp_list[perm.index(element)] = 1
        temp_list.append(label)
        dict01[key] = temp_list
    return dict01

permdict_normal_01 = permdict_to_01(permdict_normal,permissions,0)
permdict_mal_01 = permdict_to_01(permdict_mal,permissions,1)

permissions.append('label')

normal_df = pd.DataFrame.from_dict(permdict_normal_01, orient='index',columns=permissions)
mal_df = pd.DataFrame.from_dict(permdict_mal_01, orient='index',columns=permissions)
df = pd.concat([normal_df,mal_df])

df.to_csv('perm_train_temp.csv')


