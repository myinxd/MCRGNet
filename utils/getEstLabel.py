# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

"""
Final estimation
1. load all estimated labels of the corresponding dichotomous branches
2. combine to get the code
3. save as a table

Usage
=====
python3 getEstLabel <foldname> 
# Labels estimated by all of the five dichotomous branches are required. 
"""


import sys
import os
import pickle
import numpy as np

def main(argv):
    # folder for saving
    foldname = argv[1]
    if not os.path.exists(subfold):
        os.mkdir(subfold)
    # load dichotomous labels
    fnames = ['est_l1.pkl','est_l2.pkl','est_l3.pkl','est_l4.pkl','est_l5.pkl']

    # l1
    with open(os.path.join(foldname, fnames[0]), 'rb') as fp:
        dict_l1 = pickle.load(fp)

    # l2
    with open(os.path.join(foldname, fnames[1]), 'rb') as fp:
        dict_l2 = pickle.load(fp)

    # l3
    with open(os.path.join(foldname, fnames[2]), 'rb') as fp:
        dict_l3 = pickle.load(fp)

    # l4
    with open(os.path.join(foldname, fnames[3]), 'rb') as fp:
        dict_l4 = pickle.load(fp)

    # l5
    with open(os.path.join(foldname, fnames[4]), 'rb') as fp:
        dict_l5 = pickle.load(fp)

    # Judge
    numsample = len(dict_l1['label_pos'])
    codes = []
    types = []
    types_BT = []
    pos = []
    pos_l1 = dict_l1['label_pos']
    pos_l2 = dict_l2['label_pos']
    pos_l3 = dict_l3['label_pos']
    pos_l4 = dict_l4['label_pos']
    pos_l5 = dict_l5['label_pos']
    for i in range(numsample):
        c = "".join([str(dict_l1['label_est'][i]),
                     str(dict_l2['label_est'][i]),
                     str(dict_l3['label_est'][i]),
                     str(dict_l4['label_est'][i]),
                     str(dict_l5['label_est'][i])
                    ])
        if c[0] == '0':
            t = 1
            t_BT = 0
            pos.append(pos_l1[i])
        elif c[0:3] == "100":
            t = 2
            t_BT = 0
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        elif c[0:3] == "101":
            t = 3
            t_BT = 0
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        elif c[0:2] == "11" and c[3] == "0":
            t = 4
            if c[2] == "0":
                t_BT = 1;
            else:
                t_BT = 2;
            pos.append(pos_l1[i]*pos_l2[i]*pos_l4[i])
        elif c[0:2] == "11" and c[3:5] == "10":
            t = 5
            t_BT = 0
            pos.append(pos_l1[i]*pos_l2[i]*pos_l4[i]*pos_l5[i])
        elif c[0:2] == "11" and c[3:5] == "11":
            t = 6
            t_BT = 0
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i]*pos_l5[i])
        else:
            t = 9
            t_BT = 0
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i]*pos_l4[i]*pos_l5[i])
        codes.append(c)
        types.append(t)
        types_BT.append(t_BT)

    label_real = dict_l1["label_est"]
    z = dict_l1["z"]
    snvss = dict_l1['snvss']
    name = dict_l1['name']

    from pandas import DataFrame
    labeldict = {"code":codes,
                 "types":types, "types_BT":types_BT,
                 "z": z, "S_NVSS": snvss, "Name-J2000": name, "Possibility": pos,
                 "Poss_l1": pos_l1, "Poss_l2": pos_l2, "Poss_l3": pos_l3,
                 "Poss_l4": pos_l4, "Poss_l5": pos_l5}
    dframe = DataFrame(labeldict)
    dframe.to_excel("%s/EstLabel.xlsx" % (foldname))

if __name__ == "__main__":
    main(sys.argv)
