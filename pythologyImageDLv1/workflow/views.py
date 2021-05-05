from django.shortcuts import render
from django.http import HttpResponse, JsonResponse,HttpResponseRedirect

import json
import os;

from . import service;

# Create your views here.
def index(request):


	return render(request,"Workflow.html" );

def prepare(request):

	return render(request,"Prepare.html")

def removeImage(request):

	path = request.POST.get("path");

	result = service.removeImage(path);

	return JsonResponse(result);


def queryDatasource(request):

	data=service.queryDatasource();
	return JsonResponse({"result": data});

def queryRawImageList(request):
	subpath = request.POST.get("path");

	data = service.queryRawImageList(subpath);

	return JsonResponse({"result": data});

def savecrop(request):
	path = request.POST.get("path");
	w = request.POST.get("w");
	h=  request.POST.get("h");
	x = request.POST.get("x");
	y=  request.POST.get("y");
	name = request.POST.get("name");

	data = service.saveCrop(path,x,y,w,h,name);

	return JsonResponse(data);

def autocrop(request):
	path = request.POST.get("path");
	w = request.POST.get("w");
	h = request.POST.get("h");
	result = service.autocrop(path,w,h);

	return JsonResponse(result);


def saveautocrop(request):
	path = request.POST.get("path");
	crops = request.POST.get("crops");
	crops = crops.split("//,;");
	crops = [ [x.split("___")[0],x.split("___")[1].split("_") ]  for x in crops];
	cropsdict=dict();
	for i in crops:
		cropsdict[i[0]]=i[1];

	result = service.saveMultiCrops(path,cropsdict);

	return JsonResponse(result);

def queryFileList(request):

	path = request.POST.get("path");

	data = service.queryFileList(path);

	return JsonResponse({"result": data});

def getScaledImageByWH(request):
	w = request.POST.get("w");
	h=  request.POST.get("h");
	method = request.POST.get("method");
	
	path = request.POST.get("path");

	if method == "cv2":
		path = os.path.join(service.getAnnotationPath(),path);
		path = service.getImageByPath(path);
		data = service.getScaledImageByWHAndCV2(path,w,h);
	else:
		data = service.getScaledImageByWH(path,w,h);

	return JsonResponse(data);

def getFullImage(request):
	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);
	path = service.getImageByPath(path);
	data = service.getFullImage(path);

	return JsonResponse(data);

def getScaledCropedImage(request):
	path = request.POST.get("path");
	w = request.POST.get("w");
	h=  request.POST.get("h");
	x = request.POST.get("x");
	y=  request.POST.get("y");

	resizeW = request.POST.get("resizeW");
	resizeH=  request.POST.get("resizeH");

	path = os.path.join(service.getAnnotationPath(),path);
	path = service.getImageByPath(path);

	method = request.POST.get("method"); 

	if method == "cv2":
		data = service.getScaledCropedImageUseCV2(path,x,y,w,h,resizeW,resizeH);
	else:
		data = service.getScaledCropedImage(path,x,y,w,h,resizeW,resizeH);

	return JsonResponse(data);


def getRawCropedImage(request):

	path = request.POST.get("path");
	w = request.POST.get("w");
	h=  request.POST.get("h");
	x = request.POST.get("x");
	y=  request.POST.get("y");

	data = service.getRawCropedImage(path,x,y,w,h);

	return JsonResponse(data);

def queryDLModels(request):

	data = service.queryDLModels();

	return JsonResponse(data);

def aiModelPredict(request):
	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);
	path = service.getImageByPath(path);

	w = request.POST.get("w");
	h=  request.POST.get("h");
	x = request.POST.get("x");
	y=  request.POST.get("y");
	model = request.POST.get("model");

	rateW = request.POST.get("rateW");
	rateH=  request.POST.get("rateH");

	data = service.aiModelPredict(path,x,y,w,h,model,rateW,rateH);

	return JsonResponse(data);

def saveAIpredictResult(request):

	path =request.POST.get("path");

	path = os.path.join(service.getAnnotationPath(),path);

	contours =request.POST.get("contours");

	contours=contours.split(";")

	contours = [x.split(",") for x in contours];

	contours2 =[];

	for i in contours:
		eachContour = [[ int(x.split("_")[0]),int(x.split("_")[1])  ] for x in i];
		contours2.append(eachContour);
	
	contours=None;

	name =request.POST.get("name");


	result = service.saveAIpredictResult(path,name,contours2);


	return JsonResponse(result);


def checkPredictStatus(request):


	job_id = request.POST.get("job_id");
	result =service.checkPredictResult(job_id);

	return JsonResponse(result);


def queryAnnotations2(request):
	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);

	data = service.queryAnnotations2(path);

	return JsonResponse(data)

def queryAnnotations(request):
	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);

	data = service.queryAnnotations(path);

	return JsonResponse(data)

def aiclassification(request):

	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);

	anno = request.POST.get("anno");

	data = service.aiclassification(path,anno);

	return JsonResponse(data)




def saveAnnotation(request):

	path = request.POST.get("path");
	name = request.POST.get("name");
	attr = request.POST.get("attr");
	coors= request.POST.get("coors");

	path = os.path.join(service.getAnnotationPath(),path);

	coors = coors.split(";");
	coors = [x.split(",") for x in coors];

	coors = [[int(x[0]),int(x[1]) ] for x in coors ];

	
	result = service.saveAnnotation(path,name,attr,coors);

	return JsonResponse(result);


def removeAnnotation(request):
	aid = request.POST.get("aid");
	path = request.POST.get("path");
	path = os.path.join(service.getAnnotationPath(),path);
	category = request.POST.get("category");

	result = service.removeAnnotation(aid,category,path);

	return JsonResponse(result);

def queryModelInputSize(request):
	model = request.POST.get("model");

	result = service.queryModelInputSize(model);

	return JsonResponse(result);

def MakeTrainingSets(request):
	category = request.POST.get("category");
	path = request.POST.get("path");
	x = request.POST.get("x");
	y = request.POST.get("y");
	w = request.POST.get("w");
	h = request.POST.get("h");
	stepx = request.POST.get("stepx");
	stepy = request.POST.get("stepy");
	inputx = request.POST.get("inputx");
	inputy = request.POST.get("inputy");

	epochs = request.POST.get("epochs");

	dlmodel = request.POST.get("model");


	path = os.path.join(service.getAnnotationPath(),path);

	result = service.MakeTrainingSets(category,path,[x,y,w,h],[stepx,stepy],[inputx,inputy],epochs,dlmodel);
	return JsonResponse(result);



def updateAttr(request):
	attr = request.POST.get("attr");
	path = request.POST.get("path");
	annoid = request.POST.get("id");
	path = os.path.join(service.getAnnotationPath(),path);
	name = request.POST.get("name");
	result = service.updateAnnoAttr(path,name,annoid,attr);

	return JsonResponse(result);


def filterAnnotations(request):
	path = request.POST.get("path");
	name = request.POST.get("category");
	w = request.POST.get("w");
	h=  request.POST.get("h");
	x = request.POST.get("x");
	y=  request.POST.get("y");
	
	path = os.path.join(service.getAnnotationPath(),path);


	result = service.filterAnnotations(path,name,x,y,w,h);
	return JsonResponse(result);


def checkTrainingStatus(request):
	job_id = request.POST.get("job_id");
	result =service.checkTrainingStatus(job_id);

	return JsonResponse(result);



def queryImages(request):
	result = service.queryImages();
	
	return JsonResponse(result);

