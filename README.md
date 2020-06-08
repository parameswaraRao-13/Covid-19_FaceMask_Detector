# Covid-19_FaceMask_Detector

By: Parameswara rao Gutti

date: 04 Jun 2020

I have implimented model by using train_data.csv and validation_data.csv data sets to predict Covid-19 face mask. This project is organised as follows:
1. Used cafee pretrained model(transfer learning) for fask detection in a frame and extracted those pixels.
2. Implimented CNN classifier by using Imagenet VGG16 as convolutional base.
3. Detect masks in extracted pixels by using above trainrd CNN classifier model.
