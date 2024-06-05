from django.db import models

class EnergySpending(models.Model):
    hausehold = models.CharField(max_length=11)
    energy = models.FloatField()
    time = models.DateTimeField(null=True)
    
    class Meta:
        ordering = ("hausehold", )
        
    def __str__(self):
        return self.hausehold