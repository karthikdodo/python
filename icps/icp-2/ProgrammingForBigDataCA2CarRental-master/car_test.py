import unittest

from car import Car, EngineCar, PetrolCar, DieselCar, ElectricCar, HybridCar

class TestCar(unittest.TestCase):

    def setUp(self):
        self.car = Car()

    def test_default_constructor(self):
        self.assertEquals('', self.car.make)
        self.assertEquals('', self.car.model)

    def test_car_get_method(self):
        self.assertEquals('', self.car.getMake())
        self.assertEquals('', self.car.getModel())

    def test_car_set_method(self):
        self.car.setMake('Mercedes')
        self.car.setModel('e230')
        self.assertEquals('Mercedes', self.car.getMake())
        self.assertEquals('e230', self.car.getModel())

#class TestEngineCar(unittest.TestCase):
class TestEngineCar(TestCar):

    def setUp(self):
        TestCar.setUp(self)
        self.engineCar = EngineCar()

    def test_engine_default_constructor(self):
        self.assertEquals('', self.engineCar.transmission)
        
    def test_car_get_methods(self):
        self.assertEquals('', self.engineCar.getMake())
        self.assertEquals('', self.engineCar.getModel())

    def test_car_set_methods(self):
        self.engineCar.setTransmission('Automatic')
        self.assertEquals('Automatic', self.engineCar.getTransmission())
        
        
class TestPetrolCar(TestEngineCar):

    def setUp(self):
        TestEngineCar.setUp(self)
        self.petrolCar = PetrolCar()

    def test_petrol_default_constructor(self):
        self.assertEquals('', self.petrolCar.petrolType)
        
    def test_petrol_get_method(self):
        self.assertEquals('', self.petrolCar.getPetrolType())

    def test_petrol_set_method(self):
        self.petrolCar.setPetrolType('Unleaded')
        self.assertEquals('Unleaded', self.petrolCar.getPetrolType())

        
class TestDieselCar(TestEngineCar):

    def setUp(self):
        TestEngineCar.setUp(self)
        self.dieselCar = DieselCar()

    def test_diesel_default_constructor(self):
        self.assertEquals(False, self.dieselCar.turbo)
        
    def test_diesel_get_method(self):
        self.assertEquals(False, self.dieselCar.getDieselTurbo())

    def test_diesel_set_method(self):
        self.dieselCar.setDieselTurbo(True)
        self.assertEquals(True, self.dieselCar.getDieselTurbo())
        

class TestElectricCar(TestCar):

    def setUp(self):
        TestCar.setUp(self)
        self.electricCar = ElectricCar()

    def test_electric_default_constructor(self):
        self.assertEquals('', self.electricCar.fuelCells)
        
    def test_electric_get_method(self):
        self.assertEquals('', self.electricCar.getFuelCells())

    def test_electric_set_method(self):
        self.electricCar.setFuelCells('Battery')
        self.assertEquals('Battery', self.electricCar.getFuelCells())

class TestHybridCar(TestPetrolCar, TestElectricCar):

    def setUp(self):
        TestPetrolCar.setUp(self)
        TestElectricCar.setUp(self)
        self.hybridCar = HybridCar()

    def test_hybrid_default_constructor(self):
        self.assertEquals(False, self.hybridCar.regenerativeBrake)
        
    def test_hybrid_get_method(self):
        self.assertEquals(False, self.hybridCar.getHybridBrake())

    def test_hybrid_set_method(self):
        self.hybridCar.setHybridBrake(True)
        self.assertEquals(True, self.hybridCar.getHybridBrake())

if __name__ == '__main__':
    unittest.main()

