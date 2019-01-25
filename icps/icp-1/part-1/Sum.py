# an example for raw_input and int conversion

firstNo=10    #interger type
secondNo=20.0 #float type
name="UMKC"

print('welcome to',name)
print (firstNo,' plus ',secondNo,' equals ',firstNo+secondNo)


"""num1String = raw_input('Please enter an integer: ')         #python 2
num2String = raw_input('Please enter a second integer: ') """

num1String = input('Please enter an integer: ')
num2String = input('Please enter a second integer: ')

num1 = int(num1String)
num2 = int(num2String)

print ('Here is some output')


#print num1,' plus ',num2,' equals ',num1+num2   --python2
#print 'Thanks for playing'

print (num1,' plus ',num2,' equals ',num1+num2)

print ('Thanks you. END')