class Employee:
    ct=0
    def __init__(self,n,a,s,d):
        self.name=n
        self.age=a
        self.salary=s
        self.department=d
    def increment(self):
       self.__class__. ct+=1


def get_average(list):

    x=sum(list)
    n=len(list)
    print(x/n)



class FullTimeEmployee(Employee):
    def __init__(self, name, age, salary, department, status):
        Employee.__init__(self, name, age, salary, department)
        self.status = status



sal=[]
e1=Employee("karthik",22,100000,"java")

sal.append(e1.salary)
e1.increment()
e2=Employee("santosh",22,100000,"python")
sal.append(e2.salary)
e1.increment()
e3=Employee("mourya",22,200000,"all")
sal.append(e3.salary)
e1.increment()
n=e1.__class__.ct
print(n)

get_average(sal)

