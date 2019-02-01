class CarRental(object):
    
    def __init__(self):
        self.fleet = 0
        self.availablePetrolCars = []
        self.availableDieselCars = []
        self.availableElectricCars = []
        self.availableHybridCars = []

    ## setters
    def setFleet(self, fleet):
        self.fleet = fleet
    
    def setAvailablePetrolCars(self, availablePetrolCars):
        self.availablePetrolCars = availablePetrolCars
    
    def setAvailableDieselCars(self, availableDieselCars):
        self.availableDieselCars = availableDieselCars
        
    def setAvailableElectricCars(self, availableElectricCars):
        self.availableElectricCars = availableElectricCars
        
    def setAvailableHybridCars(self, availableHybridCars):
        self.availableHybridCars = availableHybridCars

    ## getters
    def getFleet(self):
        return self.fleet
    
    def getAvailablePetrolCars(self):
        return self.availablePetrolCars
        
    def getAvailableDieselCars(self):
        return self.availableDieselCars
        
    def getAvailableElectricCars(self):
        return self.availableElectricCars
        
    def getAvailableHybridCars(self):
        return self.availableHybridCars
