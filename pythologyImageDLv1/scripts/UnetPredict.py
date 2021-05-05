import sys
import os;
import json;
import math;

import tensorflow as tf
#import keras;
from tensorflow import keras;
from keras.models import load_model

import cv2;
import numpy as np;

import matplotlib.pyplot as plt;


job_dir = "./Temp";


def main():
	job_id = sys.argv[1];

	inputfile = os.path.join( job_dir,job_id+"_input.txt" );
	resfile = os.path.join( job_dir,job_id+"_data.txt" );
	logfile =os.path.join( job_dir,job_id+"_log.txt" );
	imgfile = os.path.join( job_dir,job_id+"_img.png" );

	inputarr=["modelpath","ids","cutborder","patchsize","resW","resH","cutoff"];

	cutborder=0;
	patchsize=[];
	resW=0;
	resH=0;
	cutoff=0;
	ids=[];
	modelpath="";

	index=0;

	args_arr=[];
	with open(inputfile) as f:

		for i in f:
			args_arr.append(i.strip());


	if len(args_arr) == len(inputarr):
		for i in range(len(args_arr)):

			if inputarr[i]=="ids":
				ids= args_arr[i].split(",");
				ids= [x.split("_") for x in ids];
				ids = [ [int(x[0]),int(x[1])] for x in ids];
				ids=ids;

			elif inputarr[i]=="patchsize":
				patchsize= args_arr[i].split("_");

				patchsize=[ int(patchsize[0]),int(patchsize[1])];

			elif inputarr[i] == "resW":
				resW=int(args_arr[i]);
			elif inputarr[i] == "resH":
				resH=int(args_arr[i]);
			elif inputarr[i] == "cutborder":
				cutborder=  int(args_arr[i]);
			elif inputarr[i] == "cutoff":
				cutoff=int(args_arr[i]);

			elif inputarr[i] == "modelpath":
				modelpath = args_arr[i];
			

			
	else:
		with open( logfile, "w") as f:
			f.write("error , input args not correct");
		return "error";

	

	unetmodel = tf.keras.models.load_model(modelpath);

	img = cv2.imread(imgfile);
	img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
	img =img/255.0;

	startXY = ids[0];
	endXY = ids[-1]

	final_test_img=np.zeros((img.shape[0],img.shape[1],1));
	new_test_img = np.empty((img.shape[0],img.shape[1],1));
	new_test_img_count = np.zeros((img.shape[0],img.shape[1]));


	index=0;
	for i in ids:
		temp = img[i[1]: (i[1]+patchsize[1] ), i[0] :  (i[0]+patchsize[0] ) ];
		#temp = cv2.normalize(temp, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3);
		#temp =temp/255.0;
		temp = np.asarray([temp]);

		temp = unetmodel.predict(temp);
		temp = temp[0];
		
		#matrix = cv2.morphologyEx(matrix, cv2.MORPH_CLOSE, (3,3))
		for y in range(len(temp)):
			for x in range(len(temp[y])):
				"""
				if i[1] !=startXY[1] and y<cutborder:
					continue;
				if i[0] !=startXY[0] and x <cutborder:
					continue;

				if i[0] != endXY[0] and (x+cutborder) > patchsize[0]:
					continue;
				if i[1] != endXY[1] and (y+cutborder) > patchsize[1]:
					continue;  
				"""
				val=temp[y][x];
				val =np.array(val);
				new_test_img[i[1]+y][ i[0]+x] =  new_test_img[i[1]+y][ i[0]+x]+ val;
				new_test_img_count[i[1]+y][ i[0]+x] +=1;

		finish =  math.floor( (index/float(len(ids)) )* 92);
		index+=1;
		with open( logfile, "w") as f:
			f.write(str(finish)+"%");


	for y in range(len(final_test_img)):

		if y == int(len(final_test_img)/2):
			with open( logfile, "w") as f:
				f.write("96%");

		for x in range(len(final_test_img[y])):
			final_test_img[y][x]= np.array(new_test_img[y][x])/int(new_test_img_count[y][x])

	with open( logfile, "w") as f:
		f.write("98%");
	final_test_img =  np.asarray(final_test_img*255 , dtype=np.uint8);

	final_test_img = cv2.resize(final_test_img,( resW,resH ))
	ret, binary = cv2.threshold(final_test_img,cutoff,255,cv2.THRESH_BINARY);


	#if modelpath == "DLModels/Unet_for_Mucosa/mucosa_temp3.hdf5":
	#	cv2.imwrite(binary,)


	contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
	#contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
	resContours=[];
	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		if area <100:
			continue;

		temp=[];
		for j in contours[i]:
			temp.append([int(j[0][0]),int(j[0][1]) ]);
		resContours.append({"coors":temp})
	
	with open(resfile, 'w') as f:
		json.dump(resContours, f)


	with open( logfile, "w") as f:
		f.write(str("100%"));
	
main()