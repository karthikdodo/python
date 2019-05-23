import re
first=input('Enter first name')
last=input('Enter last name')
"""print(first[::-1])
print(last[::-1])"""


print(''.join((reversed(first))))

print(''.join((reversed(last))))



a=10
b=20
print(a+b)
print(a-b)
print(a*b)
print(a/b)
x=input("Enter string:")
num=0
word=0
for i in x:
    if(i.isnumeric()):
        num+=1
    elif(i.isalpha()):
        word+=1


print(num,word)






