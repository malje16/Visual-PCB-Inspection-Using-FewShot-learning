''' 
Because this seems to keep happening I'm gonna create a library of common dataset manipulations. I'm using cocoapi and also 
cocoassistant cause they already have some of the functionality I need. 
''' 

import json 
import cv2
import pycocotools.coco as tools
import coco_assistant as assistant
import os
import shutil
import skimage.io as io 
import matplotlib.pyplot as plt
import numpy as np 
import copy
import json


def resize_img_sidelength(img, size=1000):
    if img.shape[1] > img.shape[0]:
        width = size
        scale_percent = int((width * 100) / img.shape[1])
        height = int(img.shape[0] * scale_percent / 100)
    else: 
        height = size
        scale_percent = int((height * 100) / img.shape[0])
        width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def dict_to_list(dictionary): 
    value_list =[]
    for key, value in dictionary.items():
        value_list.append(value)
    return value_list 

class dataset: 

    def __init__(self, img_dir=None, ann_file=None):
        
        self.tools_object = tools.COCO(ann_file)
        self.ann_file = ann_file
        if img_dir==None :
            self.img_dir = None
        else: 
            self.img_dir = img_dir


    def visualise_id(self, img_to_show_id):
        current_img = io.imread(os.path.join(self.img_dir, self.tools_object.loadImgs(img_to_show_id)[0]['file_name']))
        current_ann = self.tools_object.loadAnns(self.tools_object.getAnnIds(imgIds=img_to_show_id))
        plt.imshow(current_img)
        plt.axis('off')
        self.tools_object.showAnns(current_ann, draw_bbox=True)
        print("showing image with id: " + str(img_to_show_id))
        plt.show()


    def visualise_dataset(self):
        for img_to_show_id in self.tools_object.getImgIds():
            self.visualise_id(img_to_show_id)


    def visualise_img(self, img_to_show_path):
        for img_data in self.tools_object.dataset['images']:
            if img_data['file_name'] == img_to_show_path:
                self.visualise_id(img_data['id'])
    
    def visualise_filename(self, filename):
        ids = self.tools_object.getImgIds
        something = self.tools_object.loadImgs(self.tools_object.getImgIds)
        #if image_dict['filename'] == filename:
        #    self.visualise_id(image_dict['id'])
        return

    def write_coco_json(self, output_dir, data_dict = None): # data_dict should contain these keys: info licenses images annotations categories
        # prioritize input list. If no arguments are given use self.dataset 
        if data_dict == None: 
            data_dict = { 
                'info': self.tools_object.dataset['info'],
                'licenses': self.tools_object.dataset['licenses'],
                'images': self.tools_object.imgs,
                'annotations': self.tools_object.anns,
                'categories': self.tools_object.cats
            }
        data_dict['images'] = dict_to_list(data_dict['images'])
        data_dict['annotations'] = dict_to_list(data_dict['annotations'])
        data_dict['categories'] = dict_to_list(data_dict['categories'])

        #new_file_name = os.path.join(output_dir, "annotations.json")
        with open(output_dir, 'wt', encoding='UTF-8') as coco:
            json.dump({ 'info': data_dict['info'], 'licenses': data_dict['licenses'], 'images': data_dict['images'], 
            'annotations': data_dict['annotations'], 'categories': data_dict['categories']}, coco, indent=2, sort_keys=True)

        
    def crop_pixel_checker_first_axis(self, current_img, outer_list, inner_list, leniency=10, intensity_limit=10):
        '''
        checks if pixels at positions outer_list x inner_list are part of background or not. 
        background defined with the border of intensity limit
        leniency lets you disregard a number of background pixels in a row. 
        returns the index for the outer loop where the background stops
        '''
        intensity_limit = [intensity_limit, 255-intensity_limit]
        previous_pixel_clean = True
        consecutive_dirty_pixels = 0
        for i in outer_list:
            for j in inner_list:
                if previous_pixel_clean:
                    consecutive_dirty_pixels = 0
                else: 
                    previous_pixel_clean = True
                for channel in current_img[i, j]:
                    if((channel > intensity_limit[0]) and (channel < intensity_limit[1])): 
                        consecutive_dirty_pixels +=1
                        previous_pixel_clean = False
                        if(consecutive_dirty_pixels > leniency): return i
                        break

    def crop_pixel_checker_second_axis(self, current_img, outer_list, inner_list, leniency=10, intensity_limit=10):
        '''
        checks if pixels at positions outer_list x inner_list are part of background or not. 
        background defined with the border of intensity limit
        leniency lets you disregard a number of background pixels in a row. 
        returns the index for the outer loop where the background stops
        '''
        intensity_limit = [intensity_limit, 255-intensity_limit]
        previous_pixel_clean = True
        consecutive_dirty_pixels = 0
        for i in outer_list:
            for j in inner_list:
                if previous_pixel_clean:
                    consecutive_dirty_pixels = 0
                else: 
                    previous_pixel_clean = True
                for channel in current_img[j, i]:
                    if((channel > intensity_limit[0]) and (channel < intensity_limit[1])): 
                        consecutive_dirty_pixels +=1
                        previous_pixel_clean = False
                        if(consecutive_dirty_pixels > leniency): return i
                        break

    def crop_background(self, output_dir, run_blind=False):
        '''
        Uses pixel checking to find and remove background from the edges of images. 
        set run_blind to True to execute the method without visualising the results of the cropping. 

        Return: saves a new cropped version of the dataset(annotation and images), at output_dir
        '''

        min_background = 10
        new_data_dict = { 
                'info': self.tools_object.dataset['info'],
                'licenses': self.tools_object.dataset['licenses'],
                'images': self.tools_object.imgs,
                'annotations': self.tools_object.anns,
                'categories': self.tools_object.cats
            }
        if not os.path.exists(os.path.join(output_dir, 'images')):
            os.makedirs(os.path.join(output_dir, 'images'))
        for current_img_id in self.tools_object.getImgIds():
            
            current_img_dict = self.tools_object.loadImgs(current_img_id)[0] 
            print("current img id : " + str(current_img_id) + ", name of img: " + current_img_dict['file_name'])
            current_img = io.imread(os.path.join(self.img_dir, current_img_dict['file_name']))
            
            row_index = range(0, current_img.shape[0], 3)
            coloumn_index = range(0, current_img.shape[1], 3)
            first_axis_low = self.crop_pixel_checker_first_axis(current_img, row_index, coloumn_index)

            row_index = range(first_axis_low, current_img.shape[0], 3)
            second_axis_low = self.crop_pixel_checker_second_axis(current_img, coloumn_index, row_index)

            coloumn_index = range(second_axis_low, current_img.shape[1], 3)
            reverse_row_index = reversed(range(0, current_img.shape[0], 3))
            first_axis_high = self.crop_pixel_checker_first_axis(current_img, reverse_row_index, coloumn_index)

            row_index = range(first_axis_low, first_axis_high, 3)
            reverse_coloumn_index = reversed(range(0, current_img.shape[1], 3))
            second_axis_high = self.crop_pixel_checker_second_axis(current_img, reverse_coloumn_index, row_index)


            if (first_axis_low > min_background) or (second_axis_low > min_background) or ((current_img.shape[0] - first_axis_high) > min_background) or ((current_img.shape[1] - second_axis_high) > min_background):

                response = None
                while response == None:
                    cropped_img = current_img[first_axis_low:first_axis_high, second_axis_low:second_axis_high]
                    if run_blind:
                        response = "121"
                    else:
                        print("New potential crop: ")
                        cv2.imshow("Current Image", resize_img_sidelength(current_img))
                        cv2.imshow("Cropped Image", resize_img_sidelength(cropped_img))
                        #cv2.waitKey(0)
                        print("Enter \'y\' to accept crop and \'n\' to deny.")
                        response = str(cv2.waitKey(0))
                        cv2.destroyAllWindows()
                    

                    if response == "121":
                        cv2.imwrite(os.path.join(output_dir, "images", current_img_dict['file_name']), cropped_img)
                        new_data_dict['images'][current_img_id]['width'] = cropped_img.shape[1]
                        new_data_dict['images'][current_img_id]['height'] = cropped_img.shape[0]
                        for ann_id in (self.tools_object.getAnnIds(current_img_id)):
                            new_data_dict['annotations'][ann_id]['bbox'][0] -= second_axis_low
                            new_data_dict['annotations'][ann_id]['bbox'][1] -= first_axis_low
                    elif response == "110":
                        cv2.imwrite(os.path.join(output_dir, "images", current_img_dict['file_name']), current_img)
                    else: 
                        print("Did not understand input")
                        response == None
            else : 
                cv2.imwrite(os.path.join(output_dir, "images", current_img_dict['file_name']), current_img)
        self.write_coco_json(output_dir, new_data_dict)
    
    #def get_img_ids_from_filename(self, list_of_img_names)
    #    self.getImgIds

    def remove_imgs(self, list_of_img_names, output_dir):
        data_dict = { 
                'info': self.tools_object.dataset['info'],
                'licenses': self.tools_object.dataset['licenses'],
                'images': copy.copy(self.tools_object.imgs),
                'annotations': copy.copy(self.tools_object.anns),
                'categories': self.tools_object.cats
            }
        image_ids_to_remove = []
        for index in data_dict['images']:
            for img_name in list_of_img_names:
                if (data_dict['images'][index]['file_name'] == img_name):
                    image_ids_to_remove.append(data_dict['images'][index]['id'])
                    break
        ann_ids_to_remove = self.tools_object.getAnnIds(image_ids_to_remove)
        ann_ids_to_remove.sort()
        image_ids_to_remove.sort()
        ann_ids_to_remove = reversed(ann_ids_to_remove)
        img_ids_to_remove = reversed(image_ids_to_remove)
        for img_id in image_ids_to_remove:
            data_dict['images'].pop(img_id)
        for ann_id in ann_ids_to_remove:
            data_dict['annotations'].pop(ann_id)
        self.write_coco_json(output_dir, data_dict)
    
    def print_category_distribution(self):
        for cat_id in self.tools_object.getCatIds():
            print(self.tools_object.cats[cat_id]['name'] + ": " + str(len(self.tools_object.getAnnIds(catIds=cat_id))))


    def sample_dataset(self, list_of_img_names, output_dir):
        data_dict = { 
            'info': self.tools_object.dataset['info'],
            'licenses': self.tools_object.dataset['licenses'],
            'images': self.tools_object.imgs,
            'annotations': self.tools_object.anns,
            'categories': self.tools_object.cats
            }
        image_ids_to_add = []
        for index in data_dict['images']:
            for img_name in list_of_img_names:
                if (data_dict['images'][index]['file_name'] == img_name):
                    image_ids_to_add.append(data_dict['images'][index]['id'])
                    break

        new_data_dict = {
            'info': self.tools_object.dataset['info'],
            'licenses': self.tools_object.dataset['licenses'],
            'images': {},
            'annotations': {},
            'categories': self.tools_object.cats
        }
        
        ann_ids_to_add = self.tools_object.getAnnIds(image_ids_to_add)

        img_id_counter = 0
        ann_id_counter = 0
        for img_id in image_ids_to_add:
            new_data_dict['images'][img_id_counter] = data_dict['images'][img_id]
            img_id_counter +=1 
        for ann_id in ann_ids_to_add:
            new_data_dict['annotations'][ann_id_counter] = data_dict['annotations'][ann_id]
            ann_id_counter += 1

        new_data_dict['images'] = dict(new_data_dict['images'])

        self.write_coco_json(output_dir, new_data_dict)
'''
    def load_vgg(self, ann_file, coco_example):
        coco_dataset = tools.COCO(coco_example)
        coco_dict = { 
            'info': coco_dataset.dataset['info'],
            'licenses': coco_dataset.dataset['licenses'],
            'images': coco_dataset.imgs,
            'annotations': coco_dataset.anns,
            'categories': coco_dataset.cats
        }

        with open(ann_file, 'r') as f:
            dataset = json.load(f)
        data_dict = {
                'info': [],
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': []
        }
        for image in dataset: 
            new_img_dict = {
                'id':  , 
                license
            }

        return
'''