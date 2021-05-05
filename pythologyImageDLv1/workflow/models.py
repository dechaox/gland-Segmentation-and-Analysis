from django.db import models

# Create your models here.
class dataSource(models.Model):

	name = models.CharField(max_length=200,primary_key=True);
	path = models.TextField( unique=True);

	def __str__(self):
		return self.name;


class Annotation(models.Model):
	path = models.TextField( unique=False);
	name = models.CharField(max_length=200 );
	coors = models.TextField(default="" );
	annoJson=models.TextField( );

	def __str__(self):
		return self.name;