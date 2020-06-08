import xml.etree.ElementTree as ET
import csv
import pandas as pd
import glob
import os
from PIL import Image
# open a file for writing

validation_data = open('validation.csv', 'w')

# create the csv writer object
csvwriter = csv.writer(validation_data)


img_dir=r'/home/ram/PycharmProjects/covid_mask/covid-19__face_mask_detection-dataset/COVID-19/validation/images'
i=glob.glob(img_dir+"/*g")
base_dir=r'/home/ram/PycharmProjects/covid_mask/data_for_model'
if not os.path.isdir(base_dir):
     os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
     os.mkdir(train_dir)
validation_dir=os.path.join(base_dir,"validation")
if not os.path.isdir(validation_dir):
     os.mkdir(validation_dir)
c_imgs_without_mask=os.path.join(train_dir,'with_out_mask')
if not os.path.isdir(c_imgs_without_mask):
     os.mkdir(c_imgs_without_mask)
c_imgs_with_mask=os.path.join(train_dir,'with_mask')
if not os.path.isdir(c_imgs_with_mask):
     os.mkdir(c_imgs_with_mask)
c_imgs_without_mask_for_validation=os.path.join(validation_dir,'with_out_mask')
if not os.path.isdir(c_imgs_without_mask_for_validation):
     os.mkdir(c_imgs_without_mask_for_validation)
c_imgs_with_mask_for_validation=os.path.join(validation_dir,'with_mask')
if not os.path.isdir(c_imgs_with_mask_for_validation):
     os.mkdir(c_imgs_with_mask_for_validation)

l=glob.glob("/home/ram/PycharmProjects/covid_mask/covid-19__face_mask_detection-dataset/COVID-19/validation/annotations/*.xml")
count = 0
o=1
#l=glob.glob("/home/ram/PycharmProjects/covid_mask/*.xml")
for file in l:
    tree = ET.parse(file)
    root = tree.getroot()
    data_head = []
    filename = root.find('filename').tag
    data_head.append(filename)
    p = 0
    for member in root.findall('size'):
        width = member.find('width').tag
        data_head.append(width)
        height = member.find('height').tag
        data_head.append(height)
        depth = member.find('depth').tag
        data_head.append(depth)


    for member in root.findall('object'):
        try:
            p=p+1
            img = []
            if count == 0:
                name = member.find('name').tag
                data_head.append(name)
                xmin = member[4][0].tag
                data_head.append(xmin)
                ymin = member[4][1].tag
                data_head.append(ymin)
                xmax = member[4][2].tag
                data_head.append(xmax)
                ymax = member[4][3].tag
                data_head.append(ymax)

                csvwriter.writerow(data_head)
                count = count + 1

            filename = str(root.find('filename').text)
            img.append(filename)
            for mem in root.findall('size'):
                width = mem.find('width').text
                img.append(width)
                height = mem.find('height').text
                img.append(height)
                depth = mem.find('depth').text
                img.append(depth)
            name = str(member.find('name').text)
            img.append(name)
            xmin = int(member[4][0].text)
            img.append(xmin)
            ymin = int(member[4][1].text)
            img.append(ymin)
            xmax = int(member[4][2].text)
            img.append(xmax)
            ymax = int(member[4][3].text)
            img.append(ymax)

            csvwriter.writerow(img)
            f=img_dir+'/'+filename

            try:
                iii = Image.open(f)
                crop_image=iii.crop((xmin,ymin,xmax,ymax))
                new_img=crop_image.resize((224,224))
                if name=='face':
                    f_n=c_imgs_without_mask_for_validation+'/'+str(p)+filename
                    new_img.save(f_n)
                elif name=='face_mask':
                    f_n = c_imgs_with_mask_for_validation + '/' + str(p) + filename
                    new_img.save(f_n)

            except:
                print("unable to crop")



        except:
            pass

validation_data.close()









