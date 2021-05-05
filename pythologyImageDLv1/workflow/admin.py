from django.contrib import admin

# Register your models here.
from .models import dataSource, Annotation



admin.site.register(dataSource);
admin.site.register(Annotation);

