# TODO

# load image
# visualize image
# break image into chunks. 
# read Annotation (pycocotools?) 
# create annotations for the new image


#Understand how the bboxes are measured, ie what does the "height" number actually mean in relation to the matrix 
#--------------------------------------------------------------------------------------------

import cv2
import pycocotools.coco as tools
import coco_assistant as assistant
import os
import shutil
import skimage.io as io 
import matplotlib.pyplot as plt
import numpy as np
#from pycocotools import COCO


def split_img_along_axis(img, axis, minimum_img_dimension1=400):

    # split img into fragments along axis with the given dimensions. 

    # Calculate the number of possible splits by dividing the axis dimension with the desired minimum_img_dimenstion
    axis_nr_of_splits = np.floor(float(np.shape(img)[axis])/minimum_img_dimension1)

    # If the image dimensions is less than the minimum_img_dimension, let dimension stay unchanged. 
    if (axis_nr_of_splits == 0):
        axis_dimension = np.shape(img)[axis]

    # Find the actual dimension for evenly sized chunks by dividing with the nr of splits 
    else:
        axis_dimension = int(np.floor(float(np.shape(img)[axis])/axis_nr_of_splits))
    
    # Extrapolate from the calculated dimensions of the to be image fragments where to split the original image. save positions in list.
    axis_split_positions = []
    for i in range(1, int(axis_nr_of_splits)):
        axis_split_positions.append(i*axis_dimension)

    # split image at the positions along the given axis
    axis_data = np.split(img, axis_split_positions, axis)
    return(axis_data)


def split_img(img_mat, ann_dict_list, minimum_img_dimension=400):
    #split single image into fragments

    # Split along the first axis. x axis, along coloumns. Resulting in a list of smaller images.  
    first_axis_data = split_img_along_axis(img_mat, 0, minimum_img_dimension1=minimum_img_dimension)
    image_fragments = []   # The list where the final image fragments are put. 
    for img_fragment in first_axis_data: # loop through the fragments from splitting along coloumns and split along the rows. 
        second_axis_data = split_img_along_axis(img_fragment, 1, minimum_img_dimension1=minimum_img_dimension)
        for j in second_axis_data:  # Save each image fragment in image_fragments. Fragments appear in reading order. 
            image_fragments.append(j)
        #image_fragments.append(second_axis_data)
    
    #split annotations for the single image 
    new_ann_list = []   # List for the new anotations
    minimum_bbox_overlap = 20  # The number of pixels a bbox needs to be into the image to be counted. 
    for i in range(len(first_axis_data)):    # Loop through all fragments, and for each fragment transfer the annotations from the original image.
        for j in range(len(second_axis_data)):  # To the corresponding fragments. 
            for ann in ann_dict_list:   # For each fragment loop through all annotations and figure out which ones apply to the fragment 
                # Map out boundaries of the bbox from the annotation.  
                largest_coloumn_index_bbox = ann['bbox'][0] + ann['bbox'][2]
                smallest_coloumn_index_bbox = ann['bbox'][0]
                largest_row_index_bbox = ann['bbox'][1] + ann['bbox'][3]
                smallest_row_index_bbox = ann['bbox'][1]

                # Read fragment dimensions
                fragment_shape = [first_axis_data[0].shape[0], second_axis_data[0].shape[1]]

                # map how the current fragment is placed in the original image 
                largest_coloumn_index_image_fragment = fragment_shape[1] * (j+1)
                smallest_coloumn_index_image_fragment = fragment_shape[1] * j
                largest_row_index_image_fragment = fragment_shape[0] * (i+1)
                smallest_row_index_image_fragment = fragment_shape[0] * i

                # Check if the annotation is represented in the current fragment. 
                if(not((largest_coloumn_index_bbox < (smallest_coloumn_index_image_fragment + minimum_bbox_overlap)) or
                    (smallest_coloumn_index_bbox > (largest_coloumn_index_image_fragment - minimum_bbox_overlap)) or
                    (largest_row_index_bbox < (smallest_row_index_image_fragment + minimum_bbox_overlap)) or
                    (smallest_row_index_bbox > (largest_row_index_image_fragment - minimum_bbox_overlap)))):
                    
                    # If the annotation is present in the fragment convert the bbox to the fragment's format. And add a new annotation entry in new_an_list
                    bbox_coloumn = ann['bbox'][0]-(j*fragment_shape[1])
                    bbox_row = ann['bbox'][1]-(i*fragment_shape[0])
                    bbox_width = ann['bbox'][2]
                    bbox_height = ann['bbox'][3]

                    # if the bbox exceed the boundaries of the fragment, rein it in. 
                    if bbox_coloumn < 0: 
                        bbox_width = bbox_width + bbox_coloumn 
                        bbox_coloumn = 0
                    if bbox_row < 0: 
                        bbox_height = bbox_height + bbox_row 
                        bbox_row = 0
                    if (bbox_coloumn + bbox_width) > fragment_shape[1]: 
                        bbox_width = bbox_width - ((bbox_coloumn + bbox_width) - fragment_shape[1])
                        bbox_width = np.floor(bbox_width)
                    if (bbox_row + bbox_height) > fragment_shape[0]: 
                        bbox_height = bbox_height - ((bbox_row + bbox_height) - fragment_shape[0])
                        bbox_height = np.floor(bbox_height)
                    
                    # Create new annotation. id is set to -1, but they reference the imagefragment in image_id. again reading order -> left to right, top to bottom
                    new_ann = {
                        'id': -1,
                        'image_id': i*len(second_axis_data) + j,
                        'category_id': ann['category_id'],
                        'segmentation': ann['segmentation'],
                        'area': ann['area'],
                        'bbox': [bbox_coloumn, bbox_row, bbox_width, bbox_height],
                        'iscrowd': ann['iscrowd'],
                        'attributes': ann['attributes']
                    }
                    new_ann_list.append(new_ann)

    return(image_fragments, new_ann_list)

def write_coco_json(img_dict_list, ann_dict_list, categories, output_file_name):
    '''
    generate a coco-annotation-format json file for the dataset. 
    '''

    f = open(output_file_name, "w")
    #Info Section
    f.write("{\n")
    f.write("    \"info\": {\n")
    f.write("        \"year\": \"2021\",\n")
    f.write("        \"description\": \"DSLR PCB dataset partition converted to COCO-format.\"\n")
    f.write("    },\n")

    #licenses section 
    f.write("    \"licenses\": [ \n")
    f.write("        {\n")
    f.write("            \"id\": 1,\n")
    f.write("            \"url\": \"\",\n")
    f.write("            \"name\": \"Unknown\"\n")
    f.write("        }\n")
    f.write("    ],\n")

    #categories section: 
    f.write("    \"categories\": [\n")
    if categories:
        list_all_components = []
        category_comma = False
        category_counter = 0
        for cat in categories:
                    if category_comma:
                        f.write(",\n")
                    else: 
                        category_comma = True
                    #Write texts for new category.
                    f.write("        {\n")
                    f.write("            \"id\": " + str(cat['id']) + ",\n")
                    category_counter += 1 
                    f.write("            \"name\": \"" + cat['name'] + "\"\n") 
                    f.write("        }")
    else: 
        f.write("        {\n")
        f.write("            \"id\": " + "0" + ",\n")
        f.write("            \"name\": \"" + "component" + "\"\n") 
        f.write("        }")
    f.write("\n    ],\n")

    #images section. 
    image_comma = False
    f.write("    \"images\": [\n")
    for img in img_dict_list:
        if image_comma:
            f.write(",\n")
        else: 
            image_comma = True 
        f.write("        {\n")
        f.write("            \"id\": " + str(img["id"]) + ",\n")
        f.write("            \"license\": 1,\n")
        #f.write("            \"file_name\": \"" + self.pcb(i).image_path() + "\",\n") #work on non cropped images
        f.write("            \"file_name\": \"" + img["file_name"] + "\",\n")
        f.write("            \"height\": " + str(img["height"]) + ",\n")
        f.write("            \"width\": " + str(img["width"]) + "\n")
        f.write("        }")
    f.write("\n    ],\n")
    #annotations section
    f.write("    \"annotations\": [\n")
    annotation_id_counter = 0
    annotation_comma = False
    for ann in ann_dict_list:
            if annotation_comma:
                f.write(",\n")
            else: 
                annotation_comma = True
            f.write("        {\n")
            f.write("            \"id\": " + str(ann["id"]) + ",\n")
            f.write("            \"image_id\": " + str(ann['image_id']) + ",\n" )
            f.write("            \"category_id\": " + str(ann['category_id']) + ",\n")
            f.write("            \"bbox\": [\n")
            f.write("                " + str(ann['bbox'][0]) + ",\n")
            f.write("                " + str(ann['bbox'][1]) + ",\n")
            f.write("                " + str(ann['bbox'][2]) + ",\n")
            f.write("                " + str(ann['bbox'][3]) + "\n")
            f.write("            ],\n")
            f.write("            \"area\": " + str(ann['area']) + ",\n")
            f.write("            \"segmentation\": [],\n")
            f.write("            \"iscrowd\": 0\n")
            f.write("        }")
    f.write("\n    ]\n")
    f.write("}")
    f.close()

def split_dataset(ann_file_path, img_dir, output_dir, minimum_img_dimension=400):

    dataset = tools.COCO(ann_file_path)  # Load the dataset to be split 
    list_of_ann_info_dicts = dataset.loadAnns(dataset.getAnnIds()) # list of original annotation dicts

    img_id_counter = 1  #counters to keep track of ID's for the split dataset
    ann_id_counter = 1 
    new_list_of_img_dicts = [] # The new list of img and ann info that are made for the split dataset
    new_list_of_ann_dicts = []
    try: 
        shutil.rmtree(os.path.join(output_dir, "images/train"))
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        return
    os.mkdir(os.path.join(output_dir, "images/train"))

    for img_id in dataset.getImgIds(): # Loop through every image, split it, and then add it to the new split dataset. 
        current_img_dict = dataset.loadImgs(img_id)[0] # Load the image dict for the current image in the original dataset
        current_img = io.imread(os.path.join(img_dir, current_img_dict['file_name'])) # Load the image referenced by the current image dict 
        current_ann = [] # The annotations belonging to the current image. These annotations need to be split. 
        for ann_dict in list_of_ann_info_dicts: # Going through all the annotations from the original dataset
            if ann_dict['image_id'] == img_id:  # If the annotation belongs to the current image 
                current_ann.append(ann_dict)    # Add it to the annotations to be split 
        current_split = split_img(current_img, current_ann, minimum_img_dimension)  # Split up the current image, and annotations. Return list of image dict and list of annotation dicts. 

        for ann in current_split[1]:
            new_ann = {
                            'id': ann_id_counter,
                            'image_id': ann['image_id'] + img_id_counter,
                            'category_id': ann['category_id'],
                            'segmentation': ann['segmentation'],
                            'area': ann['area'],
                            'bbox': ann['bbox'],
                            'iscrowd': ann['iscrowd'],
                            'attributes': ann['attributes']
                        }
            new_list_of_ann_dicts.append(new_ann)
            ann_id_counter += 1

        naming_counter = 0
        for image_fragment in current_split[0]:
            fragment_file_name = current_img_dict['file_name'].rsplit('.', 1)[0] + "_" + str(naming_counter) + ".jpg" 
            cv2.imwrite(os.path.join(output_dir, "images/train", fragment_file_name), image_fragment)
            new_img_dict = { 
                'id': img_id_counter,
                'width': image_fragment.shape[1],
                'height': image_fragment.shape[0],
                'file_name': fragment_file_name,
                'licens': current_img_dict['license'],
                'flickr_url': current_img_dict['flickr_url'],
                'coco_url': current_img_dict['coco_url'],
                'date_captured': current_img_dict['date_captured']
            }
            new_list_of_img_dicts.append(new_img_dict)
            naming_counter += 1
            img_id_counter += 1
    
    write_coco_json(new_list_of_img_dicts, new_list_of_ann_dicts, dataset.loadCats(dataset.getCatIds()), os.path.join(output_dir, "annotations", "train.json"))
    return new_list_of_img_dicts, new_list_of_ann_dicts



#splitting stuff 

annotation_dir = "/home/maltenj/projects/dataset_manipulation/cropped_dataset"
ann_file_path = "/home/maltenj/projects/dataset_manipulation/cropped_dataset/annotations.json"
img_dir = "/home/maltenj/projects/dataset_manipulation/cropped_dataset/images"
output_dir = "/home/maltenj/projects/dataset_manipulation/data_partitioned"

split_dicts = split_dataset(ann_file_path, img_dir, output_dir, minimum_img_dimension=1300)


#Testing if the split worked
new_ann_dir = os.path.join(output_dir, "annotations")
new_img_dir =os.path.join(output_dir, "images", "train")

dataset = tools.COCO(os.path.join(new_ann_dir, "train.json"))

# Visualise images with pycoco tools

for image_id_to_show in dataset.getImgIds():
    current_img = io.imread(os.path.join(new_img_dir, dataset.loadImgs(image_id_to_show)[0]['file_name']))
    current_ann = dataset.loadAnns(dataset.getAnnIds(image_id_to_show))

    #Visualize single image
    plt.imshow(current_img)
    plt.axis('off')
    dataset.showAnns(current_ann, draw_bbox=True)
    print(image_id_to_show)
    plt.show()