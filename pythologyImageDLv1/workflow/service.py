import os, sys, csv,json,datetime,time,math,scipy.stats, collections;
import re

import numpy as np;
import scipy as sp;
import math;

import sklearn;
import base64;

import copy
import random
import operator;

import multiprocessing as mp

from threading import  Thread

import subprocess
from subprocess import Popen, PIPE,STDOUT

import shutil;
import cv2;

import openslide
import numpy

import matplotlib.pyplot as plt;

import tensorflow as tf
#import keras;
from tensorflow import keras;
from keras.models import load_model


from .models import *



job_dir = "./Temp"


"""
ann = Annotation.objects.filter(path="/home/ubuntu/imagedata/2018-04-24/DSS-28 mutiple layer cutting-118.svs").values("coors","id");

test= {};

for i in ann:
	coor = i["coors"];
	coor = coor.split(";")
	coor =[x.split(",") for x in coor];
	coor = [[int(x[0])-677,int(x[1])-7216 ] for x in coor ]
	id = i["id"];
	save =True;
	for x in coor:
		if x[0] <=0 or x[0] >= 7000:
			save=False;
			break;

		if x[1] <=0 or x[1] >=7000:
			save = False;

			break;

	if save:
		test[str(id)]={"coors":coor,"attr":""}


with open("gland_anno.json", "w") as f:
	json.dump(test, f)

"""





def queryDatasource():

	datasource = dataSource.objects.all().values();
	return list(datasource);


def getAnnotationPath():
	path ="";
	with open('config') as f:
		for i in f:
			i=i.strip();
			if i != "":
				i = i.split("	");
				if len(i) ==2:
					if i[0] == "annotationPath":
						path  = i[1]
	path = path.strip();

	return path;


def getRawImagePath(subpath):
	path ="";
	with open('config') as f:
		for i in f:
			i=i.strip();
			if i != "":
				i = i.split("	");
				if len(i) ==2:
					if i[0] == "rawdataPath":
						path  = i[1]
	path = os.path.join(path,subpath);

	return path;

def queryRawImageList(subpath):
	path = getRawImagePath(subpath);

	files = os.listdir(path);
	patharr=[];
	for i in files:
		newpath = os.path.join(path,i);
		
		isdir =os.path.isdir(newpath)
		patharr.append({"name":i,"path":newpath,"isdir":isdir});
	return patharr;

def queryFileList(path):

	subpath = os.listdir(path);

	patharr=[];
	for i in subpath:
		newpath = os.path.join(path,i);
		
		isdir =os.path.isdir(newpath)
		patharr.append({"name":i,"path":newpath,"isdir":isdir});
	return patharr;


def getScaledImageByWH(path,w,h):

	w =int(w);
	h=int(h);
	source = openslide.open_slide(path);

	raw_size = source.dimensions;

	img=source.get_thumbnail((w,h))
	
	img_size = img.size;
	img = list(img.getdata());

	scaleX = raw_size[0]/img_size[0];
	scaleY = raw_size[1]/img_size[1];

	return {"pixels":img,"size":img_size,"scale":[scaleX,scaleY],"rawSize":raw_size};


def rgb2hex(r, g, b):
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def getScaledImageByWHAndCV2(path,w,h):

	image = cv2.imread(path);
	imageshape = image.shape;

	raw_size=[imageshape[1],imageshape[0]]

	scaleX = raw_size[0]/float(w);
	scaleY = raw_size[1]/float(h);

	if scaleX <1:
		scaleX=1;
	if scaleY <1:
		scaleY=1;

	scale = scaleX;
	if scale  < scaleY:
		scale = scaleY;

	scale= float(scale)

	image = cv2.resize(image,( int(raw_size[0]/scale), int(raw_size[1]/scale)) );

	

	img_size =[ image.shape[1],image.shape[0]];

	image2=[];
	for row in range(len(image)):
		temp =[];
		for col in range(len(image[row])):
			hexval = image[row][col];
			hexval = rgb2hex(hexval[0],hexval[1],hexval[2]).upper();
			temp.append(hexval)

		image2.append(temp)

	#image = image.tolist();

	return {"image":image2,"size":img_size,"scale":[raw_size[0]/img_size[0],raw_size[1]/img_size[1]],"rawSize":raw_size}; 



def getImageByPath(path):
	image = os.path.join(path,"crop.png");
	return image;

def getFullImage(path):

	image = cv2.imread(path);
	
	image = image.tolist()

	return {"pixels":image}


def getScaledCropedImageUseCV2(path,x,y,w,h,resizeW,resizeH):
	w =int(float(w));
	h=int(float(h));
	x=int(float(x));
	y=int(float(y));
	resizeH=int(float(resizeH));
	resizeW=int(float(resizeW));
	img = cv2.imread(path);

	img = img[y:y+h,x:x+w]

	raw_size = [img.shape[1],img.shape[0]];

	rate1 = raw_size[0]/resizeW;
	rate2 = raw_size[1]/resizeH;
	rate=0;
	if rate1 > rate2:
		rate = rate1;
	else:
		rate = rate2;
	if rate <1:
		rate=1;

	new_resizeW = math.floor(raw_size[0]/rate);
	new_resizeH = math.floor(raw_size[1]/rate);

	img = cv2.resize(img,(new_resizeW,new_resizeH));

	img_size = [float(img.shape[1]),float(img.shape[0]) ];

	scaleX = raw_size[0]/img_size[0];
	scaleY = raw_size[1]/img_size[1];

	image2=[];
	for row in range(len(img)):
		temp =[];
		for col in range(len(img[row])):
			hexval = img[row][col];
			hexval = rgb2hex(hexval[0],hexval[1],hexval[2]).upper();
			temp.append(hexval)

		image2.append(temp)


	return {"image":image2,"size":img_size,"scale":[scaleX,scaleY],"rawRegion":[x,y,raw_size[0],raw_size[1]]}; 

def getScaledCropedImage(path,x,y,w,h,resizeW,resizeH):

	w =int(float(w));
	h=int(float(h));
	x=int(float(x))
	y=int(float(y));
	resizeH=int(float(resizeH));
	resizeW=int(float(resizeW));

	source = openslide.open_slide(path);
	img =source.read_region( (x,y),0,(w,h));
	raw_size = img.size;
	img =  np.array( img );

	

	img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

	rate1 = raw_size[0]/resizeW;
	rate2 = raw_size[1]/resizeH;
	rate=0;
	if rate1 > rate2:
		rate = rate1;
	else:
		rate = rate2;


	if rate <1:
		rate=1;

	new_resizeW = math.floor(raw_size[0]/rate);
	new_resizeH = math.floor(raw_size[1]/rate);

	img = cv2.resize(img,(new_resizeW,new_resizeH));

	img_size = [float(img.shape[1]),float(img.shape[0]) ]

	scaleX = raw_size[0]/img_size[0];
	scaleY = raw_size[1]/img_size[1];

	img = img.tolist();


	return {"pixels":img,"size":img_size,"scale":[scaleX,scaleY],"rawRegion":[x,y,raw_size[0],raw_size[1]]}; 


def saveMultiCrops(path,crops):
	annoPath = getAnnotationPath();
	dirs = os.listdir(annoPath);
	for i in crops:
		if i in dirs:
			return {"result":"file name exists"};

	for i in crops:
		result = saveCrop(path,int(crops[i][0]),int(crops[i][1]),int(crops[i][2]),int(crops[i][3]),i );

	return {"result":"success"}


def saveCrop(path,x,y,w,h,name):
	annoPath = getAnnotationPath();

	dirs = os.listdir(annoPath);

	if name in dirs:

		return {"result":"file name exists"}

	cropedImg = getRawCropedImage2(path,x,y,w,h);
	newpath = os.path.join(annoPath,name);

	os.mkdir(newpath);
	imgpath = os.path.join(newpath,"crop.png");
	infoPath = os.path.join(newpath,"info.txt");
	infostr="path\t"+path+"\n";
	infostr+="x\t"+str(x)+"\n";
	infostr+="y\t"+str(y)+"\n";

	with open(infoPath,"w") as f:
		f.write(infostr);

	cv2.imwrite(imgpath,cropedImg);
	
	return {"result":"success"}

def autocrop(path,w,h):
	w =int(w);
	h=int(h);
	result=getScaledImageByWH(path,w,h);
	
	pixels =result["pixels"];
	size = result["size"];
	rawSize = result["rawSize"];
	rate = [rawSize[0]/size[0],rawSize[1]/size[1]];
	areaRate = rate[0]*rate[1];

	pixels2=[];
	for i in pixels:
		r, g, b = i[0], i[1], i[2];
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b;
		gray = gray;
		pixels2.append(gray)
	

	pixels = np.array(pixels2)
	
	pixels = np.reshape(pixels,(size[1],size[0]));

	kernel=(5,5)
	pixels = cv2.blur(pixels,kernel)
	#
	#pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
	#pixels = cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, kernel)

	ret, binary = cv2.threshold(pixels,210,255,cv2.THRESH_BINARY);
	

	#binary = cv2.dilate(binary,kernel,iterations = 5)
	#binary = cv2.erode(binary,kernel,iterations = 5)

	#binary = cv2.erode(binary,kernel,iterations = 8)
	#binary = cv2.dilate(binary,kernel,iterations = 8)
	binary = np.asarray(binary,dtype=np.uint8)

	#cv2.RETR_TREE   RETR_EXTERNAL
	contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
	boxs = [];

	for c in contours:
		rect = cv2.boundingRect(c)
		area = cv2.contourArea(c)
		if area*areaRate < 1000*1000:
			continue;
		if list(rect) == [0,0,size[0],size[1]] :
			continue;
		boxs.append(list(rect))
	return {"boxs":boxs};


def getRawCropedImage2(path,x,y,w,h):
	w =int(float(w));
	h=int(float(h));
	x=int(float(x));
	y=int(float(y));
	source = openslide.open_slide(path);
	img =source.read_region( (x,y),0,(w,h));
	raw_size = img.size;
	img =  np.array( img );

	img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB);

	return img

def getRawCropedImage(path,x,y,w,h):

	w =int(float(w));
	h=int(float(h));
	x=int(float(x))
	y=int(float(y));


	source = openslide.open_slide(path);
	img =source.read_region( (x,y),0,(w,h));

	raw_size = img.size;
	img =  np.array( img );

	img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB);
	img = img.tolist();

	return {"pixels":img,"size":raw_size};



def queryDLModels():

	dlmodelsdir = "DLModels";

	dlmodels=os.listdir(dlmodelsdir);

	dlinfo=dict();
	for i in dlmodels:
		infofile = os.path.join(dlmodelsdir,i,"info.json")
		info = json.load( open(infofile) );
		dlinfo[i]=info;
	
	return dlinfo;

def queryModelInputSize(modelName):
	
	modelpath = getModelPathByName(modelName);

	nnmodel = tf.keras.models.load_model(modelpath);
	input_shape=nnmodel.layers[0].input_shape[0];

	return {"shape":input_shape[1:3]};



def getDLModel(model_path):
	dlmodelsdir = "DLModels";
	modelpath=os.path.join(dlmodelsdir,model_path);

	file2 = os.listdir(modelpath);
	for f in file2:
		if f.endswith("hdf5"):
			modelpath=os.path.join(modelpath,f);
			break;


	return modelpath;


def getPatchs(inputshape,step,w,h):
	x_count = math.ceil((w-int(inputshape[0]) )/float(step[0]))+1
	y_count = math.ceil((h-int(inputshape[1]) )/float(step[1]))+1;
	x_array=[];
	y_array=[];
	for i in range(x_count):
		xindex= int(step[0])*i;

		if xindex+int(inputshape[0]) > w:
			xindex = w-int(inputshape[0]);

		x_array.append(xindex)

	for i in range(y_count):
		yindex= int(step[1])*i;
		if yindex+int(inputshape[1]) > h:
			yindex = h-int(inputshape[1]);

		y_array.append(yindex)

	ids=[];
	for i in x_array:
		for j in y_array:
			ids.append( [i,j] );

	return ids;


def getPredictedImage(img,job_id,unetmodel,ids,cutborder,patchsize,resW,resH,cutoff):

	resfile = os.path.join( job_dir,job_id+"_data.txt" );
	logfile =os.path.join( job_dir,job_id+"_log.txt" );

	img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
	img =img/255.0;


	startXY = ids[0];
	endXY = ids[-1]

	final_test_img=np.zeros((img.shape[0],img.shape[1],1));
	new_test_img = np.empty((img.shape[0],img.shape[1],1));
	new_test_img_count = np.zeros((img.shape[0],img.shape[1]));

	index=0;
	for i in ids:
		print(i)
		temp = img[i[1]: (i[1]+patchsize[1] ), i[0] :  (i[0]+patchsize[0] ) ];
		temp = cv2.normalize(temp, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3);
		temp =temp/255.0;
		temp = np.asarray([temp]);

		temp = unetmodel.predict(temp);
		temp = temp[0];

		#matrix = cv2.morphologyEx(matrix, cv2.MORPH_CLOSE, (3,3))
		for y in range(len(temp)):
			for x in range(len(temp[y])):
				if i[1] !=startXY[1] and y<cutborder:
					continue;
				if i[0] !=startXY[0] and x <cutborder:
					continue;

				if i[0] != endXY[0] and (x+cutborder) > patchsize[0]:
					continue;
				if i[1] != endXY[1] and (y+cutborder) > patchsize[1]:
					continue;  

				val=temp[y][x];
				val =np.array(val);
				new_test_img[i[1]+y][ i[0]+x] =  new_test_img[i[1]+y][ i[0]+x]+ val;
				new_test_img_count[i[1]+y][ i[0]+x] +=1;

		finish =  math.floor( (index/float(len(ids)) )* 98);
		index+=1;
		with open( logfile, "w") as f:
			f.write(str(finish)+"%");

		print(i)

	for y in range(len(final_test_img)):
		for x in range(len(final_test_img[y])):
			final_test_img[y][x]= np.array(new_test_img[y][x])/int(new_test_img_count[y][x])
	final_test_img =  np.asarray(final_test_img*255 , dtype=np.uint8);

	final_test_img = cv2.resize(final_test_img,( resW,resH ))
	ret, binary = cv2.threshold(final_test_img,cutoff,255,cv2.THRESH_BINARY);

	contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
	
	resContours=[];
	for i in range(len(contours)):
		temp=[];
		for j in contours[i]:
			temp.append([int(j[0][0]),int(j[0][1]) ]);
		resContours.append({"coors":temp})
	
	with open(resfile, 'w') as f:
		json.dump(resContours, f)


	with open( logfile, "w") as f:
		f.write(str("100%"));
	
	return "";


def saveAIpredictResult(path,name,contours):

	filesInPath = os.listdir(path);

	targetName= name+"_airesult.txt";

	if targetName in filesInPath:

		return {"result":"Name already exists"}; 

	else:
		targetPath = os.path.join(path,targetName);

		data={};
		for i in range(len(contours)):
			aid = str(i);
			attr="";
			coors = contours[i];
			data[aid]={"coors":coors,"attr":""};

		with open(targetPath,"w") as f:
			json.dump(data, f)

		return {"result":"success"}; 

	pass;


def aiModelPredict(path,x,y,w,h,model_path,rateW,rateH):
	#keras.backend.clear_session();
	modelpath = getDLModel(model_path);

	nnmodel = tf.keras.models.load_model(modelpath);
	
	#nnmodel.compile(optimizer = "adam",loss="binary_crossentropy",metrics=["acc"]);

	input_shape=nnmodel.layers[0].input_shape[0];

	rateW = float(str(rateW));
	rateH = float(str(rateH));


	w =int(float(w));
	h=int(float(h));
	x=int(float(x));
	y=int(float(y));

	img = cv2.imread(path);
	img = img[y:y+h,x:x+w];



	if modelpath == "DLModels/Unet_for_Mucosa/mucosa_temp3.hdf5":
		step=(400,400);
		#img_ori_shape=img.shape;
		#new_shape = (img.shape[0],img.shape[1])
		#img =cv2.resize(img,new_shape);

	else:
		#step=(128,128);
		step=(256,256);


	#cv2.imwrite("xxx.png",img);

	raw_size = [img.shape[1],img.shape[0]];
	patchsize=input_shape[1:3];

	
	ids = getPatchs(patchsize,step,w,h);

	idsstr =[];
	for i in ids:
		idsstr.append(str(i[0])+"_"+str(i[1]) );

	idsstr=",".join(idsstr);


	#cutborder = 50;
	cutborder = 0;#32;
	cutoff = 255*0.78;
	cutoff = int(cutoff)
	
	now = datetime.datetime.now()
	job_id = now.strftime("Predict_%Y%m%d__%H%M_%S_%s")
	job_id+="_"+"".join(random.sample("abcdefghijklmpoqrestuvwxyz",10));


	resW = int(w/rateW);
	resH = int(h/rateH);

	imgfile = os.path.join( job_dir,job_id+"_img.png" );

	cv2.imwrite(imgfile,img)

	inputfile = os.path.join( job_dir,job_id+"_input.txt" );
	inputstr="";
	inputstr+=modelpath+"\n";
	inputstr+=idsstr+"\n";
	inputstr+=str(cutborder)+"\n";
	inputstr+=str(patchsize[0])+"_"+str(patchsize[1])+"\n";
	inputstr+=str(resW)+"\n";
	inputstr+=str(resH)+"\n";
	inputstr+=str(cutoff);

	with open(inputfile,"w") as f:
		f.write(inputstr)



	logfile = os.path.join( job_dir,job_id+"_log.txt" );
	with open(logfile,"w") as f:
		f.write("0%");

	t = Thread(target=runUnetKerasPredict,args=(job_id,));
	t.start();
	#p = mp.Process(target=runUnetKerasPredict,args=(img,job_id,nnmodel,ids,cutborder,patchsize,resW,resH,cutoff) )
	#getPredictedImage(img,job_id,nnmodel,ids,cutborder,patchsize,resW,resH,cutoff);
	#p = mp.Process(target=getPredictedImage,args=(img,job_id,nnmodel,ids,cutborder,patchsize,cutoff) )
	#p.start();

	return  {"job_id":job_id};


def runUnetKerasPredict(job_id):
	
	script = "./scripts/UnetPredict.py"
	command = "python "+script+" "+job_id;
	subprocess.Popen(command, shell=True, stdin=PIPE, stdout=PIPE,stderr=STDOUT)
	#print(p.stdout.read() )
	return;

def checkPredictResult(job_id):

	job_date = job_id.replace("Predict_","").split("__")[0];
	job_date=int(job_date);

	jobs = os.listdir(job_dir);
	for i in jobs:
		if i.startswith(job_id):
			pass;

		elif i.startswith("Predict_"):
			idate = i.replace("Predict_","").split("__")[0];
			idate = int(idate)
			if job_date - idate >2:
				os.remove(os.path.join(job_dir,i));

	resfile = os.path.join( job_dir,job_id+"_data.txt" );
	logfile =os.path.join( job_dir,job_id+"_log.txt" );
	imgfile = os.path.join( job_dir,job_id+"_img.png" );

	inputfile = os.path.join( job_dir,job_id+"_input.txt" );

	msg=""
	try:
		with open(logfile) as f:
			msg = next(f);
			msg = msg.strip();
	except:
		msg="";

	if msg == "100%":
		#,"contours":resContours
		with open(resfile) as f:
			data = json.load(f);

			#os.remove(resfile);
			os.remove(logfile);
			os.remove(imgfile);
			os.remove(inputfile);

			return {"msg":msg,"contours":data} 

	else:
		return {"msg":msg} 


def splitImage(img,size,resize):

	imageshape = img.shape;
	img_x=imageshape[1];
	img_y=imageshape[0];
	img_array=[];
	index_array=[];

	indexY=0;
	while True:
		#loop y
		if indexY >=  img_y :
			break;

		if indexY + size[1] >img_y:
			
			indexY=img_y-size[1];

		if indexY <0:
			indexY=0

		indexX=0;
		while True:
			#loop x;

			if indexX >= (img_x ):
				break;

			if indexX + size[0] >img_x:
				indexX=img_x-size[0];

			if indexX <0:
				indexX=0


			index=[indexX,indexY];
			sub = img[indexY:indexY+size[1],indexX:indexX+size[0],:];
			

			sub = cv2.resize(sub,resize);


			img_array.append(sub);
			index_array.append(index);

			indexX+=size[0];


		indexY+=size[1];


	


	return index_array,np.asarray(img_array)

modelFile = "./ClassificationModels/classModel1.hdf5";
classModel = tf.keras.models.load_model(modelFile);
	
def aiclassification(path,anno,target="gland"):

	#keras.backend.clear_session();

	fullimg = cv2.imread(os.path.join(path,"crop.png"))
	annotation= os.path.join(path,anno);

	annotation = json.load(open(annotation));

	inputshape=(512,512);

	input_img=[];
	input_ids=[];

	for gland in annotation:
		coors=annotation[gland]["coors"]
		attr =annotation[gland]["attr"].strip().lower();
		
		label="normal";
		if attr.startswith("atropic"):
			label = "atropic";
			
		elif attr.startswith("proliferation"):
			label = "proliferation";
		elif attr.startswith("normal"):
			label = "normal";
			
		x_max=coors[0][0];
		x_min=coors[0][0];
		y_max= coors[0][1];
		y_min= coors[0][1];

		for i in coors:
			x = i[0]
			y = i[1];
			
			if x > x_max:
				x_max=x;
				
			if x < x_min:
				x_min = x
				
			if y > y_max:
				y_max=y;
				
			if y < y_min:
				y_min = y;
		
		x_min -=10;
		x_max +=10;
		y_min -=10;
		y_max +=10;


		img = fullimg[y_min:y_max,x_min:x_max]
		
		mask = np.zeros((img.shape[0],img.shape[1]))
		
		newcoor=[];
		
		for i in coors:
			newcoor.append( np.asarray([i[0]-x_min,i[1]-y_min]) )
		
		newcoor=np.asarray(newcoor)
		  
		mask=cv2.fillConvexPoly(mask, newcoor, 1);

		expand=6
		maskshape = mask.shape
		
		mask = cv2.resize(mask, (maskshape[1]+2*expand, maskshape[0]+2*expand ) )
		
		mask = mask[expand:expand+maskshape[0],expand:expand+maskshape[1]]
		
		mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
		
		mask = (mask!=0)
		
		
		final = np.zeros_like(img);
		
		for i in range(len(final)):
			for j in range(len(final[i])):
				if mask[i][j]:
					final[i][j]=img[i][j];


		input_img.append(cv2.resize(final,inputshape));
		input_ids.append(gland);


	#do predict
	
	input_img=np.asarray(input_img);
	input_img = input_img.astype('float')
	input_img /= 255.0;

	result= classModel.predict(input_img)

	disease=["normal","atropic","proliferation"];

	result=np.round(result,3);

	for i in range(len(result)):
		aid = input_ids[i];
		annotation[aid]["predict"]=result[i].tolist();

	return {"data":annotation,"label":disease}
	


def queryAnnotations2(path):

	files = os.listdir(path);
	res = dict();
	for i in files:
		if i.endswith("_anno.json") or i.endswith("_airesult.txt"):

			res[i.replace("_anno.json","").replace("_airesult.txt","")]=i;

	return res;

def queryAnnotations(path):
	
	files = os.listdir(path);
	res = dict();
	for i in files:
		if i.endswith("_anno.json"):
			res[i.replace("_anno.json","")]=i;

	return res;

	
def saveAnnotation(path,name,attr,coors):

	path = os.path.join(path,name+"_anno.json");

	anno = {};
	if os.path.exists(path):
		with open(path) as f:
			anno = json.load( f );


	annid = random.randint(0,100);
	annid = str(annid);

	while True:
		if annid in anno:
			annid =int(annid)+1;
			annid = str(annid);

		else:
			break;

	anno[annid]={"coors":coors,"attr":attr};

	with open(path,"w") as f:
		json.dump(anno, f)


	return {"status":"success","aid":annid};



def removeAnnotation(aid,category,path):
	
	status="failed";
	annoFile = os.path.join(path,category+"_anno.json");
	anno = {};
	if os.path.exists(annoFile):
		with open(annoFile) as f:
			anno = json.load( f );

	aid = str(aid).strip();

	anno2={};
	for i in anno:
		if str(i) == aid:
			status="success";
		else:
			anno2[i]=anno[i];


	if len(anno2.keys()) ==0:

		os.remove(annoFile);
	else:

		with open(annoFile,"w") as f:
			json.dump(anno2, f)

	return {"status":status};

def filterAnnotations(path,name,x,y,w,h):
	x0=int(x);
	y0=int(y);

	x1=int(x)+int(w);
	y1=int(y)+int(h);

	annoFile = os.path.join(path,name+"_anno.json");

	if not os.path.exists(annoFile):
		return {"result":[]};


	anno = json.load(open(annoFile) );

	result=[];

	for i in anno:
		coors=anno[i]["coors"];
		inregion=False;
		for j in coors:
			if j[0] > x0 and j[0] < x1 and j[1]>y0 and j[1]<y1:
				inregion = True;
				break;
		if inregion:
			result.append({"id":i,"coors":coors,"detail":anno[i]["attr"]})


	return {"result":result};
	

def getModelPathByName(modelName):
	dlmodelsdir = "DLModels";
	modelpath=os.path.join(dlmodelsdir,modelName);

	file2 = os.listdir(modelpath);
	for f in file2:
		if f.endswith("hdf5"):
			modelpath=os.path.join(modelpath,f);
			break;

	return modelpath;

def getModelLogByName(modelName):
	dlmodelsdir = "DLModels";
	logpath=os.path.join(dlmodelsdir,modelName,"log.txt");

	
	return logpath;


def MakeTrainingSets(category,path,crop,step,inputshape,epochs,modelName):

	modelpath = getModelPathByName(modelName);

	category=category.lower();

	w =int(float(crop[2]));
	h=int(float(crop[3]));
	x=int(float(crop[0]))
	y=int(float(crop[1]));


	imgpath = getImageByPath(path);

	img = cv2.imread(imgpath);
	img = img[y:y+h,x:x+w];
	mask = np.zeros((img.shape[0],img.shape[1]))

	ann = filterAnnotations(path,category,x,y,w,h);
	ann = ann["result"];

	coors=[];
	for i in ann:
		tempcoor=[];
		for c in i["coors"]:
			tempcoor.append([c[0]-x,c[1]-y])
		coors.append( np.asarray(tempcoor))

	coors = np.asarray(coors);

	mask=cv2.fillPoly(mask, coors, 1);

	x_count = math.ceil((w-int(inputshape[0]) )/float(step[0]))+1
	y_count = math.ceil((h-int(inputshape[1]) )/float(step[1]))+1;

	x_array=[];
	y_array=[];
	for i in range(x_count):
		xindex= int(step[0])*i;
		
		if xindex+int(inputshape[0]) > w:
			xindex = w-int(inputshape[0]);

		x_array.append(xindex)

	for i in range(y_count):
		yindex= int(step[1])*i;
		
		if yindex+int(inputshape[1]) > h:
			yindex = h-int(inputshape[1]);

		y_array.append(yindex)

	ids=[];
	for i in x_array:
		for j in y_array:
			ids.append( str(i)+"_"+str(j) );

	ids=",".join(ids);

	now = datetime.datetime.now()
	job_id = now.strftime("Train_%Y%m%d__%H%M_%S_%s")
	job_id+="_"+"".join(random.sample("abcdefghijklmpoqrestuvwxyz",10));


	maskFile = os.path.join( job_dir,job_id+"_mask.png" );
	imgFile =os.path.join( job_dir,job_id+"_img.png" );
	inputFile = os.path.join(job_dir,job_id+"_input.txt")


	inputstr="";
	inputstr+=modelpath+"\n";
	inputstr+=ids+"\n";
	inputstr+=str(inputshape[0])+"\n"
	inputstr+=str(inputshape[1])+"\n"
	inputstr+=str(epochs)

	with open(inputFile,"w") as f:
		f.write(inputstr);

	cv2.imwrite(imgFile,img);
	cv2.imwrite(maskFile,mask);


	t = Thread(target=runUnetKerasTraining,args=(job_id,));
	t.start();
	
	
	model_log = getModelLogByName(modelName);

	with open(model_log,"a") as f:
		f.write(path+" "+str(epochs)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n" );


	return {"job_id":job_id}

def runUnetKerasTraining(job_id):

	script = "./scripts/UnetTraining.py"
	command = "python "+script+" "+job_id;
	subprocess.Popen(command, shell=True, stdin=PIPE, stdout=PIPE,stderr=STDOUT)

	#res=p.stdout.read();
	#print(res)
	
	return;


def updateAnnoAttr(path,name,annid,attr):

	attr=attr.strip();
	path = os.path.join(path,name+"_anno.json");

	with open(path) as f:
		anno = json.load( f );
	annid=str(annid);
	anno[annid]["attr"]=attr;

	with open(path,"w") as f:
		json.dump(anno, f)

	return {"status":"success"}


def checkTrainingStatus(job_id):
	job_date = job_id.replace("Train_","").split("__")[0];
	job_date=int(job_date);

	jobs = os.listdir(job_dir);
	for i in jobs:
		if i.startswith(job_id):
			pass;

		elif i.startswith("Train_"):
			idate = i.replace("Train_","").split("__")[0];
			idate = int(idate)
			if job_date - idate >5:
				os.remove(os.path.join(job_dir,i));

	
	logfile =os.path.join( job_dir,job_id+"_log.txt" );
	imgfile = os.path.join( job_dir,job_id+"_img.png" );
	maskfile = os.path.join( job_dir,job_id+"_mask.png" );
	inputfile = os.path.join( job_dir,job_id+"_input.txt" );
	hdf5file = os.path.join( job_dir,job_id+"_check.hdf5" );

	msg=""

	try:
		with open(logfile) as f:
			msg = next(f);
			msg = msg.strip();
	except:
		msg="";

	if msg.upper() =="START":

		return {"status":"working","result":""}
	elif msg.upper().split("__")[-1] != "DONE":

		return {"status":msg.upper().split("__")[-1] ,"result":msg.upper().split("__")[0]};

	elif msg.upper().split("__")[-1] == "DONE":

		os.remove(logfile);
		os.remove(imgfile);
		os.remove(maskfile);
		os.remove(inputfile);
		os.remove(hdf5file);
		return {"status":"done","result":msg.upper().split("__")[0]};


	return "";

def removeImage(path):
	annoPath = getAnnotationPath();
	path = os.path.join(annoPath,path);
	shutil.rmtree(path)


	return {"status":"success"}

def queryImages():
	annoPath = getAnnotationPath();

	dirs = os.listdir(annoPath);
	dirs.sort();
	data=dict();

	dirs1 = [];
	dirs2=[];

	for i in dirs:
		path = os.path.join(annoPath,i);
		files = os.listdir(path);
		
		annotedFiles=[];
		for f in files:
			if f.endswith("_anno.json"):
				annotedFiles.append(f.replace("_anno.json",""));

		if len(annotedFiles) >0:
			dirs1.append(i)

		else:
			dirs2.append(i)

	for i in dirs1+dirs2:
		path = os.path.join(annoPath,i);
		files = os.listdir(path);
		
		annotedFiles=[];
		predictedFiles=[];
		for f in files:
			if f.endswith("_anno.json"):
				annotedFiles.append(f.replace("_anno.json",""));

			if f.endswith("_airesult.txt"):
				predictedFiles.append(f.replace("_airesult.txt",""));

		data[i]={"annoted":annotedFiles,"airesult":predictedFiles}

	return data;



def training(a):
	print(a)


def dataGenerate():

	pass;




class DataGen(keras.utils.Sequence):
	def __init__(self,ids,imgpath,crop,step,batch_size=10,image_size=(256,256)):
		self.ids=ids;
		
		self.batch_size =batch_size;
		self.image_size = image_size;

	def __load__(self,id_name):
		pass;



