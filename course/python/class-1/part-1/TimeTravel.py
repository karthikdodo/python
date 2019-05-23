# prob5: TimeTravel
import math
s = input('enter speed(percentage of speed of light): ')
perc = float(s)
factor = 100/math.sqrt(10000 - perc*perc)
c = 299792458
w = 70000*factor
t1 = 4.3/factor
t2 = 6.0/factor
t3 = 309/factor
t4 = 2000000/factor
print ('weight: ', w)
print ('Alpha centauri: ', t1)
print ('Bernard\'s Star: ', t2)
print ('Betelgeuse: ', t3)
print ('Andromeda: ', t4)