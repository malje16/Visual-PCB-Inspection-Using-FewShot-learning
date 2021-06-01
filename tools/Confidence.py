import numpy as np
import scipy.stats
import json
import os


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def get_data(filename):
    f = open(filename, 'r')
    return json.load(f)

def triplet(namestring):
    name_list = []
    name_list.append(namestring + "1")
    name_list.append(namestring + "2")
    name_list.append(namestring + "3")
    return name_list

base = "/home/maltenj/FrustratinglyFSOD/checkpoints/Danchell_Experiment2/"
end = "inference/res_final.json"

list_list = []
print_categories = False
#list_list.append(triplet("FsDet_2shot"))
#list_list.append(triplet("FsDet_Anchor1_2shot"))
#list_list.append(triplet("FsDet_Anchor2_2shot"))  # This one has become fairly standard
#list_list.append(triplet("FsDet_Anchor3_2shot"))
#list_list.append(triplet("FsDet_10shot"))
#list_list.append(triplet("COCO_FsDet_2shot"))
#list_list.append(triplet("COCO_FsDet_10shot"))
#list_list.append(triplet("FsDet_lr_2shot"))
#list_list.append(triplet("FsDet_lr_10shot"))
#list_list.append(triplet("FsDet_baseAspect_2shot"))
#list_list.append(triplet("FsDet_baseAspect1_2shot"))
#list_list.append(triplet("FsDet_Aspect_2shot"))
#list_list.append(triplet("FsDet_bareAugment_2shot"))
#list_list.append(triplet("FsDet_moderateAugment_2shot"))
#list_list.append(triplet("FsDet_unfreeze1_2shot"))
#list_list.append(triplet("FsDet_unfreeze2_2shot"))
list_list.append(triplet("Test_2shot"))
list_list.append(triplet("Test_10shot"))
list_list.append(triplet("Test_COCO_2shot"))
list_list.append(triplet("Test_COCO_10shot"))





for list in list_list:
    
    dict= {"AP": [], 'AP50': [], 'AP75': [], 'AP-CAP R': [], 'AP-CAP S': [], 'AP-DSUB': [], 'AP-FUSE': [], 'AP-LED': [], 'AP-PIN': [], 'AP-SCHOTTKY': []}
    for name in list:
        path = os.path.join(base, name, end)
        data = get_data(path)
        dict['AP'].append(data['bbox']['AP'])
        dict['AP50'].append(data['bbox']['AP50'])
        dict['AP75'].append(data['bbox']['AP75'])
        if print_categories:
            dict['AP-CAP R'].append(data['bbox']['AP-CAP R'])
            dict['AP-CAP S'].append(data['bbox']['AP-CAP S'])
            dict['AP-DSUB'].append(data['bbox']['AP-DSUB'])
            dict['AP-FUSE'].append(data['bbox']['AP-FUSE'])
            dict['AP-LED'].append(data['bbox']['AP-LED'])
            dict['AP-PIN'].append(data['bbox']['AP-PIN'])
            dict['AP-SCHOTTKY'].append(data['bbox']['AP-SCHOTTKY'])

    print("\n" + list[0] + ':')
    print("AP: ")
    ap = mean_confidence_interval(dict['AP'])
    ap = [round(ap[0],1), round(ap[1],1)]
    print(ap)
    print("AP50: ")
    ap = mean_confidence_interval(dict['AP50'])
    ap = [round(ap[0],1), round(ap[1],1)]
    print(ap)
    print("AP75: ")
    ap = mean_confidence_interval(dict['AP75'])
    ap = [round(ap[0],1), round(ap[1],1)]
    print(ap)
    if print_categories:
        print("AP-CAP R: ")
        ap = mean_confidence_interval(dict['AP-CAP R'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-CAP S: ")
        ap = mean_confidence_interval(dict['AP-CAP S'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-DSUB: ")
        ap = mean_confidence_interval(dict['AP-DSUB'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-FUSE: ")
        ap = mean_confidence_interval(dict['AP-FUSE'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-LED: ")
        ap = mean_confidence_interval(dict['AP-LED'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-PIN: ")
        ap = mean_confidence_interval(dict['AP-PIN'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)
        print("AP-SCHOTTKY: ")
        ap = mean_confidence_interval(dict['AP-SCHOTTKY'])
        ap = [round(ap[0],1), round(ap[1],1)]
        print(ap)