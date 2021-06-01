''' 

I have problems. 

This program is gonna try to use the library to do all the things i wanna do.
sequence of events: 
I have the dataset in 4 pieces, 3 fics and one pcb_dslr 

OK 

1. Merge
2. remove stragglers and re-merge
3. Crop
#4. partition, I will attempt to do this through Detectron2 instead. 
5. remove a class 
6. split into training and test

''' 

import pycocotools.coco as tools
from coco_assistant import COCO_Assistant
import coco_manipulation_library as coco_lib
import os
import cv2
import shutil


def merge():
    imgdir_all = os.path.join(os.getcwd(), 'all_images')

    anndir_all = os.path.join(os.getcwd(), 'all_annotations')

    cas = COCO_Assistant(imgdir_all, anndir_all)
    cas.merge(merge_images=True)  #should save the merged stuff in results/merge
    # This worked, i think. I uploaded it to cvat and it looked good. 


def stragglers():
    #okay this next bit is hard to do in a single file execution since it involves me taking out two images and 
    #re-annotating them. They are currently saved in /rotating_stragglers/, and I have a simple program in straggler_merge that performs a, well, 
    #a merge of full dataset without stragglers and the stragglers. So. This next bit will just produce the full dataset without stragglers
    # and save it in straggler merge. You can then go to straggler merge and run the merge file and get the straggler included dataset- 

    img_dir = "/home/maltenj/projects/dataset_manipulation/results/merged/images"
    ann_file = "/home/maltenj/projects/dataset_manipulation/results/merged/annotations/merged.json"

    merged_dataset = coco_lib.dataset(img_dir=img_dir, ann_file=ann_file)
    imgs_to_remove = ["pcb_116.jpg", "pcb_148.jpg", "s27_front.jpg", "s28_front.jpg", "s29_front.jpg"]
    dataset_without_stragglers_path = "/home/maltenj/projects/dataset_manipulation/straggler_merge/all_annotations"
    merged_dataset.remove_imgs(list_of_img_names=imgs_to_remove, output_dir=dataset_without_stragglers_path)
    # note that the annotation file will be called annotations.json. You will need to change the name to match the image directory Also 
    # you will need to update the images in stragglers. 
    # Im adding image s27 s28 and s29, because they are not cropped properly. Really I should resize all the fics images, because they are 
    # way to big, but that might mean re-annotating. I know detectron does a resize maybe I can utilize that 


def cropped():
    # Now I'm going to run the crop. 
    fixed_img_dir = "/home/maltenj/projects/dataset_manipulation/straggler_merge/results/merged/images"
    fixed_ann_file = "/home/maltenj/projects/dataset_manipulation/straggler_merge/results/merged/annotations/merged.json"
    crop_dir = "/home/maltenj/projects/dataset_manipulation/cropped_dataset"

    fixed_dataset = coco_lib.dataset(img_dir=fixed_img_dir, ann_file=fixed_ann_file) 

    fixed_dataset.crop_background(run_blind=True, output_dir=crop_dir)
    #uploaded to cvat and the crop looks okay. Looked through all images s3_back (if i recall correctly), had shifted boxes
    # not sure what caused it, but i fixed it manually in cvat. Also the colours are wrong, looks lige a RGB/BGR swap. I will 
    # have to check the colours are correct when i load it into detectron. Seems to be caused by the usage of skimage to read images. 
    # skimage loads as RGB and OpenCV assumes BGR.
    # I will have to download the set and fix the colours. 
    # But in general: the crop is done, just download the task: cropped, from cvat.org


# extra segment to download and convert from BGR to RGB so these images look okay.
def rgb_convert():
    old_image_dir = "/home/maltenj/projects/dataset_manipulation/task_cropped-2021_04_10_21_37_11-coco 1.0/images"
    new_image_dir = "/home/maltenj/projects/dataset_manipulation/task_cropped-2021_04_10_21_37_11-coco 1.0/images_rgb"

    for filename in os.listdir(old_image_dir):
        old_image = cv2.imread(os.path.join(old_image_dir, filename))
        new_image = cv2.cvtColor(old_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(new_image_dir, filename), new_image)

# ran python cocosplit.py -s 0.8 /home/maltenj/projects/dataset_manipulation/task_cropped-2021_04_10_21_37_11-coco1.0/annotations/instances_default.json train.json test.json
# it split stuf i moved the train and test annotation to datasets. I will read that shit and move the images with if I can. 
def moving_imgs_to_split():
    train_ann_file = "/home/maltenj/datasets/fics_pcb_merged/train.json"
    test_ann_file = "/home/maltenj/datasets/fics_pcb_merged/test.json"
    image_source = "/home/maltenj/projects/dataset_manipulation/task_cropped-2021_04_10_21_37_11-coco1.0/images_rgb"
    train_img_target = "/home/maltenj/datasets/fics_pcb_merged/train"
    test_img_target = "/home/maltenj/datasets/fics_pcb_merged/test"

    afd = os.listdir(image_source)
    #train segment
    train_object = tools.COCO(train_ann_file)
    for info_dict in train_object.loadImgs(train_object.getImgIds()):
        img = cv2.imread(os.path.join(image_source, info_dict["file_name"]))
        cv2.imwrite(os.path.join(train_img_target, info_dict['file_name']), img)

    #test segment
    test_object = tools.COCO(test_ann_file)
    for info_dict in test_object.loadImgs(test_object.getImgIds()):
        img = cv2.imread(os.path.join(image_source, info_dict["file_name"]))
        cv2.imwrite(os.path.join(test_img_target, info_dict['file_name']), img)

def remove_classes():
    # I want to remove classes with to few instances, namely the yellow capacitors. Cuz there are like 5.  
    ann_dir = "/home/maltenj/projects/dataset_manipulation/remove_classes/annotations"
    ann_file = "/home/maltenj/projects/dataset_manipulation/remove_classes/annotations/train.json"
    img_dir ="/home/maltenj/projects/dataset_manipulation/remove_classes/images/train"
    img_dir_dir = "/home/maltenj/projects/dataset_manipulation/remove_classes/images"
    new_dataset = coco_lib.dataset(img_dir=img_dir, ann_file=ann_file)
    new_dataset.print_category_distribution()

    list_of_cats_to_remove = ["capacitor_electrolytic_yellow_standing", "capacitor_electrolytic_yellow_lying", "capacitor_electrolytic_green_lying"]
    cas = COCO_Assistant(img_dir_dir, ann_dir)
    #cas.remove_cat(interactive=False, jc="train.json", rcats=list_of_cats_to_remove)
    cas.remove_cat(interactive=False, jc="test.json", rcats=list_of_cats_to_remove)

def print_categories():
    train_ann_file = "/home/maltenj/datasets/fics_pcb_merged/annotations/train.json"
    test_ann_file = "/home/maltenj/datasets/fics_pcb_merged/annotations/test.json"
    train_img_dir = "/home/maltenj/datasets/fics_pcb_merged/train"
    test_img_dir = "/home/maltenj/datasets/fics_pcb_merged/test"

    all_ann_file = "/home/maltenj/datasets/fics_pcb_merged/annotations/all.json"

    trainset = coco_lib.dataset(ann_file=train_ann_file, img_dir=train_img_dir)
    testset = coco_lib.dataset(ann_file=test_ann_file, img_dir=test_img_dir)
    set = coco_lib.dataset(ann_file=all_ann_file)
    set.print_category_distribution()

# I want to make a couple (maybe 3) specific Danchell datasets to truly test the accuraccy of My new model. I need... A couple for each instance. 
# I want one in 3 different copies with 1 image. 
# And one in 3 copyes with 3 images. 
# We could do another one with 8 images in 3 copies. 
# All said the function need to extract a new very small dataset from a bigger one. And I think I want to manually pick the images. 

def create_chosen_dataset():
    output_dir = "/home/maltenj/datasets/HighResDanchellExperiment/annotations"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    full_danchell_ann = '/home/maltenj/datasets/HighResDanchellExperiment/test/_annotations.coco.json'
    full_danchell_img = '/home/maltenj/datasets/HighResDanchellExperiment/test'

    full_danchell_dataset = coco_lib.dataset(img_dir = full_danchell_img , ann_file = full_danchell_ann)
    
    img_names_2shot1 = ['Image101_jpg.rf.647cba7791ca6b5333fb5eaa91c96ee0.jpg']
    full_danchell_dataset.sample_dataset(img_names_2shot1, output_dir + "/train_2shot1.json")
    full_danchell_dataset.remove_imgs(img_names_2shot1, output_dir + "/test_2shot1.json")

    img_names_2shot2 = ['Image102_jpg.rf.fa05476be23fdf2f0112befc2ff0a3c4.jpg']
    full_danchell_dataset.sample_dataset(img_names_2shot2, output_dir + "/train_2shot2.json")
    full_danchell_dataset.remove_imgs(img_names_2shot2, output_dir + "/test_2shot2.json")

    img_names_2shot3 = ['Image104_jpg.rf.98d6fcb7f28847d8548b5c9e0a4d907f.jpg']
    full_danchell_dataset.sample_dataset(img_names_2shot3, output_dir + "/train_2shot3.json")
    full_danchell_dataset.remove_imgs(img_names_2shot3, output_dir + "/test_2shot3.json")

    img_names_10shot1 = [
        "Image68_jpg.rf.b95ba2a2503cb0ac764379c792bfab31.jpg",
        "Image97_jpg.rf.a656ff1f9b7bb8ac50a9d8f4fca3e4b9.jpg",
        "Image134_jpg.rf.839c963b9fb20d1901608fb36cdae730.jpg",
        "Image135_jpg.rf.2ac0f773d81d55872c04919f1790256c.jpg",
        "Image154_jpg.rf.c65d5b41ebb8827d89c7f5698c03c3d7.jpg"]
    full_danchell_dataset.sample_dataset(img_names_10shot1, output_dir + "/train_10shot1.json")
    full_danchell_dataset.remove_imgs(img_names_10shot1, output_dir + "/test_10shot1.json")
    
    img_names_10shot2 = [
        "Image167_jpg.rf.49bcbe918647f87586f494be8b40db2b.jpg",
        "Image18_jpg.rf.1bd02bbd553eefe8e055e1dfa6882613.jpg",
        "Image93_jpg.rf.3ad36915a1710ebbe70312f162be0dfd.jpg", 
        "Image116_jpg.rf.2ba5c47dcffd1525e07fe7195b9be23a.jpg",
        "Image112_jpg.rf.0b57b6f44200a9e926b32ee108522d99.jpg"]
    full_danchell_dataset.sample_dataset(img_names_10shot2, output_dir + "/train_10shot2.json")
    full_danchell_dataset.remove_imgs(img_names_10shot2, output_dir + "/test_10shot2.json")

    img_names_10shot3 = [
        "Image94_jpg.rf.60d8ee18fadac4345031533253495304.jpg",
        "Image1_jpg.rf.8069cf0e2839e828e1edc2f9d61b568d.jpg"
        "Image113_jpg.rf.45f713eed72284f42ecc8487800d0b9c.jpg", 
        "Image121_jpg.rf.dc2a83b11339447f2056bd01d70dee57.jpg",
        "Image107_jpg.rf.18d6959eccd8c11d7b5b7d760484f64e.jpg"]
    full_danchell_dataset.sample_dataset(img_names_10shot3, output_dir + "/train_10shot3.json")
    full_danchell_dataset.remove_imgs(img_names_10shot3, output_dir + "/test_10shot3.json")

#create_chosen_dataset()
#print_categories()
def visualize_merged_img():
    img = "/home/maltenj/datasets/fics_pcb_merged/images"
    ann = "/home/maltenj/datasets/fics_pcb_merged/annotations/all.json"

    merged_dataset = coco_lib.dataset(img_dir=img, ann_file=ann)
    merged_dataset.visualise_img("pcb_125.jpg") 
    merged_dataset.visualise_img("s31_front.jpg") 
    

def visualize_danchell_img():
    img = "/home/maltenj/datasets/HighResDanchellExperiment/test"
    ann = "/home/maltenj/datasets/HighResDanchellExperiment/test/_annotations.coco.json"

    merged_dataset = coco_lib.dataset(img_dir=img, ann_file=ann)
    merged_dataset.visualise_img("Image108_jpg.rf.d41d95201e3781f6d5fae25801b10fdf.jpg") 

def cropping_initial():
    full_dataset_ann = "/home/maltenj/projects/dataset_manipulation/straggler_merge/results/merged/annotations/merged.json"
    full_dataset_img = "/home/maltenj/projects/dataset_manipulation/straggler_merge/results/merged/images"


    new_dataset = coco_lib.dataset(img_dir=full_dataset_img, ann_file=full_dataset_ann)
    new_dataset.crop_background("/home/maltenj/projects/dataset_manipulation/cropped_dataset", run_blind=True)

def remove_stragglers(): 
    merged_ann = "/home/maltenj/projects/dataset_manipulation/merged/annotations/merged.json"
    merged_img = "/home/maltenj/projects/dataset_manipulation/merged/images"
    test_folder = "/home/maltenj/projects/dataset_manipulation/test_folder"

    new_dataset = coco_lib.dataset(img_dir=merged_img, ann_file=merged_ann)

    imgs_to_remove = ["pcb_116.jpg", "pcb_148.jpg"]
    new_dataset.remove_imgs(imgs_to_remove, test_folder)
    straggles_removed = "/home/maltenj/projects/dataset_manipulation/test_folder/annotations.json"

    fresh_dataset = coco_lib.dataset(img_dir=merged_img, ann_file=straggles_removed)
    fresh_dataset.visualise_dataset()


def rotate(img, angle):
    rows,cols, depth = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv2.warpAffine(img,M,(cols,rows))

def rotate_stragglers(): 
    '''
    rotate some img in dataset that are at odd angles. ie 45 degrees. 
    instances: pcb_116
    instances: pcb_148
    '''


    img_dir = "/home/maltenj/projects/dataset_manipulation/merged/images"
    output_dir = "/home/maltenj/projects/dataset_manipulation/rotating_stragglers/"

    img_116 = cv2.imread(os.path.join(img_dir, "pcb_116.jpg"))
    img_148 = cv2.imread(os.path.join(img_dir, "pcb_148.jpg"))



    img_116_rot = rotate(img_116, 28)
    img_148_rot = rotate(img_148, 322)
    cv2.imshow("148 original", coco_lib.resize_img_sidelength(img_148))
    cv2.imshow("148 rotated", coco_lib.resize_img_sidelength(img_148_rot))

    cv2.imshow("116 original", coco_lib.resize_img_sidelength(img_116))
    cv2.imshow("116 rotated", coco_lib.resize_img_sidelength(img_116_rot))

    response = str(cv2.waitKey(0))

    cv2.imwrite(os.path.join(output_dir, "pcb_116_rotated.jpg"), img_116_rot)
    cv2.imwrite(os.path.join(output_dir, "pcb_148_rotated.jpg"), img_148_rot)

def the_merging():
    imgdir_fics_1_9 = os.path.join(os.getcwd(), 'fics_pcb_1-9/images_jpg')
    imgdir_fics_10_17 = os.path.join(os.getcwd(), 'fics_pcb_10-17/images_jpg')
    imgdir_fics_18_31 = os.path.join(os.getcwd(), 'fics_pcb_18-31/images_jpg')
    imgdir_all = os.path.join(os.getcwd(), 'all_images')

    anndir_fics_1_9 = os.path.join(os.getcwd(), 'fics_pcb_1-9/annotations')
    anndir_fics_10_17 = os.path.join(os.getcwd(), 'fics_pcb_10-17/annotations')
    anndir_fics_18_31 = os.path.join(os.getcwd(), 'fics_pcb_18-31/annotations')
    anndir_all = os.path.join(os.getcwd(), 'all_annotations')

    img_list = [ imgdir_fics_1_9, imgdir_fics_10_17, imgdir_fics_18_31 ]
    ann_list = [ anndir_fics_1_9, anndir_fics_10_17, anndir_fics_18_31 ]

    cas = COCO_Assistant(imgdir_all, anndir_all)
    cas.merge(merge_images=True)



visualize_merged_img()


# I'm gonna try to make a tiny 1 image dataset and train a model that massively overfits to see Problems with my process. 
# can probably just use cocosplit and try to get it to a single image. 
# I copied a 'fresh' version of the pcb_dslr_fics_merged dataset from the datasets folder. 
# python cocosplit.py -s 0.8 /home/maltenj/projects/dataset_manipulation/task_cropped-2021_04_10_21_37_11-coco1.0/annotations/instances_default.json train.json test.json

#python cocosplit.py -s 0.99 /home/maltenj/projects/dataset_manipulation/fics_pcb_merged/annotations/train.json junk.json tinyset.json


'''
ann_3shot1 = "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_3shot1.json"
danchel_imgs = "/home/maltenj/datasets/HighResDanchellExperiment/test"
dataset_3shot1_train = coco_lib.dataset(img_dir = danchel_imgs , ann_file = ann_3shot1)

dataset_3shot1_train.visualise_dataset()
'''