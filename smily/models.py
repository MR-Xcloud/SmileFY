from django.db import models

# Create your models here.
class ware(models.Model):
    file1 = models.ImageField(upload_to='file1')
    file2 = models.URLField()
    

    