from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('prepare', views.prepare, name='prepare'),

    path("queryDatasource",views.queryDatasource),

    path("queryRawImageList",views.queryRawImageList),
    path("savecrop",views.savecrop),

    path("autocrop",views.autocrop),
    path("saveautocrop",views.saveautocrop),

    path("checkPredictStatus",views.checkPredictStatus),

    path("queryFileList",views.queryFileList),
    path("getScaledImageByWH",views.getScaledImageByWH),
    path("getScaledCropedImage",views.getScaledCropedImage),
    path("getRawCropedImage",views.getRawCropedImage),
    path("queryDLModels",views.queryDLModels),
    path("aiModelPredict",views.aiModelPredict),

    path("queryAnnotations",views.queryAnnotations),
    path("saveAnnotation",views.saveAnnotation),
    path("removeAnnotation",views.removeAnnotation),

    path("queryModelInputSize",views.queryModelInputSize),
    path("filterAnnotations",views.filterAnnotations),
    path("MakeTrainingSets",views.MakeTrainingSets),

    path("removeImage",views.removeImage),

    path("checkTrainingStatus",views.checkTrainingStatus),
    path("updateAttr",views.updateAttr),
    path('queryImages', views.queryImages ),
    path("getFullImage",views.getFullImage),


    path("saveAIpredictResult",views.saveAIpredictResult),
    path("queryAnnotations2",views.queryAnnotations2),
    path("aiclassification",views.aiclassification),
]