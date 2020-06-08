import xml.etree.ElementTree as ET
import csv
import pandas as pd
import glob

# open a file for writing

train_data = open('train_data.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(train_data)
l=glob.glob("/home/ram/PycharmProjects/covid_mask/covid-19__face_mask_detection-dataset/COVID-19/training/annotations/*.xml")
count = 0
#l=glob.glob("/home/ram/PycharmProjects/covid_mask/*.xml")
for file in l:
    tree = ET.parse(file)
    root = tree.getroot()
    data_head = []
    filename = root.find('filename').tag
    data_head.append(filename)
    for member in root.findall('size'):
        width = member.find('width').tag
        data_head.append(width)
        height = member.find('height').tag
        data_head.append(height)
        depth = member.find('depth').tag
        data_head.append(depth)


    for member in root.findall('object'):
        try:
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

            filename = root.find('filename').text
            img.append(filename)
            for mem in root.findall('size'):
                width = mem.find('width').text
                img.append(width)
                height = mem.find('height').text
                img.append(height)
                depth = mem.find('depth').text
                img.append(depth)
            name = member.find('name').text
            img.append(name)
            xmin = member[4][0].text
            img.append(xmin)
            ymin = member[4][1].text
            img.append(ymin)
            xmax = member[4][2].text
            img.append(xmax)
            ymax = member[4][3].text
            img.append(ymax)
            csvwriter.writerow(img)
        except:
            pass
train_data.close()









