import unittest

from customer import Customer

class TestCustomer(unittest.TestCase):

    def setUp(self):
        self.customer = Customer()

    def test_default_constructor(self):
        self.assertEquals('', self.customer.name)
        self.assertEquals('', self.customer.licence)
        self.assertEquals(0, self.customer.hirePeriod)
        self.assertEquals(0.00, self.customer.hireCharge)

    def test_customer_get_methods(self):
        self.assertEquals('', self.customer.getName())
        self.assertEquals('', self.customer.getLicence())
        self.assertEquals(0, self.customer.getHirePeriod())
        self.assertEquals(0.00, self.customer.getHireCharge())

    def test_customer_set_methods(self):
        self.customer.setName('Paddy')
        self.assertEquals('Paddy', self.customer.getName())
        self.customer.setName('IRE161215')
        self.assertEquals('IRE161215', self.customer.getName())
        self.customer.setHirePeriod(14)
        self.assertEquals(14, self.customer.getHirePeriod())
        self.customer.setHireCharge(250.00)
        self.assertEquals(250.00, self.customer.getHireCharge())


if __name__ == '__main__':
    unittest.main()

