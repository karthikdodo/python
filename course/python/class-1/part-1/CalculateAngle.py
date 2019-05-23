# prob3: TurtleGraphicsAngles
import math
import turtle
print ('Provide two points A and B, get acute angle OAB where O is (0, 0)')
x1 = float(input('enter x-coordinate of first point: '))
y1 = float(input('enter y-coordinate of first point: '))
x2 = float(input('enter x-coordinate of second point: '))
y2 = float(input('enter y-coordinate of second point: '))

if x1 == x2:
	theta = 0
else:
	m1 = y1/x1
	m2 = (y2 - y1)/(x2 - x1)
	x = abs(m1 - m2)
	if m1*m2 == -1:
		theta = 90
	else:
		theta = math.atan(x/(1 + m1*m2))*180/math.pi
print (theta, ' degrees')
turtle.goto(0,0)
turtle.pendown()
turtle.goto(x1, y1)
turtle.goto(x2, y2)
turtle.done()