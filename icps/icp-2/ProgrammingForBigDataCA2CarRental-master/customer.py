class Customer(object):
    def __init__(self):
        self.name = ''
        self.licence = ''
        self.hirePeriod = 0
        self.hireCharge = 0.00

    ## setters
    def setName(self, name):
        self.name = name
        
    def setLicence(self, licence):
        self.licence = licence
    
    def setHirePeriod(self, hirePeriod):
        self.hirePeriod = hirePeriod
        
    def setHireCharge(self, hireCharge):
        self.hireCharge = hireCharge
     
    ## getters
    def getName(self):
        return self.name
	
    def getLicence(self):
        return self.licence
    
    def getHirePeriod(self):
        return self.hirePeriod
        
    def getHireCharge(self):
        return self.hireCharge