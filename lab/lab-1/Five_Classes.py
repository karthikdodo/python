#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:52:30 2019

@author: karthikchowdary
"""

class Person:
    
    def __init__(self,n,a,add):
        self.name=n
        self.age=a
        self.address=add
        
        
""" Person Class is created it can be inherited by Passenger and also Employee """
class Passenger(Person):
     
    def __init__(self,n,a,add,luggweight):
         Person.__init__(self,n,a,add)
         self.luggage_weight=luggweight
         
    def gettraveldate(self):
        print("24th of April")
    def getluggage(self):
        print(self.luggage_weight)
""" Passenger is a class extending Person class """       
 
class Employee(Person):
    
    def __init__(self,n,a,add,idnumber):
         Person.__init__(self,n,a,add)
         self.id=idnumber
         
    def getjoindate(self):
        print("10th of February")
     
    def getid(self):
        print(self.id)    
  
""" Employee is a class extending Person class """       
class Flight():
    fno=0
    def __init__(self,fno):
        self.flight=fno
    
    def getflight(self):
        print(self.fno)

""" flight is a class"""  
        
class Pilot(Person, Flight):
    def __init__(self,n,a,add,fno,id):
        Person.__init__(self,n,a,add)
        Flight.__init__(self,fno)
        self.id=id
        
    def getpilotid(self):
        print(self.id)
        
""" Multiple Inheritance Pilot class extends Person and Flight """  
        



pass1=Passenger("karthik",22,"india",50)
pass1.gettraveldate()
pass1.getluggage()


emp=Employee("mourya",22,"usa",16252361)
emp.getid()
emp.getjoindate()


pilot=Pilot("santy",22,"india",1665,15118)
pilot.getpilotid()
    
