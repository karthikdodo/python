class Car(object):

    def __init__(self):
        self.make = ''
        self.model = ''
        self.transmission = ''
        self.petrolType = ''
        self.turbo = ''
        self.fuelCells = ''
        self.regenerativeBrake = ''

    ## setters
    def setMake(self, make):
        self.make = make
        
    def setModel(self, model):
        self.model = model

    ## getters
    def getMake(self):
        return self.make
        
    def getModel(self):
        return self.model
	

class EngineCar(Car):
    def __init__(self):
        Car.__init__(self)
        self.transmission = ''
    
    def setTransmission(self, transmission):
        self.transmission = transmission

    def getTransmission(self):
        return self.transmission

        
class PetrolCar(EngineCar):
    def __init__(self):
        EngineCar.__init__(self)
        self.petrolType = ''
    
    def setPetrolType(self, petrolType):
        self.petrolType = petrolType

    def getPetrolType(self):
        return self.petrolType

        
class DieselCar(EngineCar):
    def __init__(self):
        EngineCar.__init__(self)
        self.turbo = False
        
    def setDieselTurbo(self, turbo):
        self.turbo = turbo

    def getDieselTurbo(self):
        return self.turbo

        
class ElectricCar(Car):
    def __init__(self):
        Car.__init__(self)
        self.fuelCells = ''
        
    def setFuelCells(self, fuelCells):
        self.fuelCells = fuelCells

    def getFuelCells(self):
        return self.fuelCells


class HybridCar(PetrolCar, ElectricCar):
    def __init__(self):
        PetrolCar.__init__(self)
        ElectricCar.__init__(self)
        self.regenerativeBrake = False
    
    def setHybridBrake(self, regenerativeBrake):
        self.regenerativeBrake = regenerativeBrake

    def getHybridBrake(self):
        return self.regenerativeBrake
