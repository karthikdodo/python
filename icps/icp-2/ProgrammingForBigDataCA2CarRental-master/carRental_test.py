import unittest

from carRental import CarRental
from car import Car, EngineCar, PetrolCar, DieselCar, ElectricCar, HybridCar

class TestCarRental(unittest.TestCase):

    def setUp(self):
        self.carRental = CarRental()

    def test_default_constructor(self):
        self.assertEquals(0, self.carRental.fleet)
        self.assertEquals([], self.carRental.availablePetrolCars)
        self.assertEquals([], self.carRental.availableDieselCars)
        self.assertEquals([], self.carRental.availableElectricCars)
        self.assertEquals([], self.carRental.availableHybridCars)
        
    def test_car_rental_get_methods(self):
        self.assertEquals(0, self.carRental.getFleet())
        self.assertEquals([], self.carRental.getAvailablePetrolCars())
        self.assertEquals([], self.carRental.getAvailableDieselCars())
        self.assertEquals([], self.carRental.getAvailableElectricCars())
        self.assertEquals([], self.carRental.getAvailableHybridCars())

    def test_car_rental_set_methods(self):
        self.carRental.setFleet(40)
        self.assertEquals(40, self.carRental.getFleet())
        petrolCar = PetrolCar()
        dieselCar = DieselCar()
        electricCar = ElectricCar()
        hybridCar = HybridCar()
        self.carRental.setAvailablePetrolCars([petrolCar, petrolCar])
        self.assertEquals([petrolCar, petrolCar], self.carRental.getAvailablePetrolCars())
        self.carRental.setAvailableDieselCars([dieselCar, dieselCar, dieselCar])
        self.assertEquals([dieselCar, dieselCar, dieselCar], self.carRental.getAvailableDieselCars())
        self.carRental.setAvailableElectricCars([electricCar, electricCar])
        self.assertEquals([electricCar, electricCar], self.carRental.getAvailableElectricCars())
        self.carRental.setAvailableHybridCars([hybridCar, hybridCar])
        self.assertEquals([hybridCar, hybridCar], self.carRental.getAvailableHybridCars())


if __name__ == '__main__':
    unittest.main()
