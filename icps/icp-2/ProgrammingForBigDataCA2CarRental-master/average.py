n=eval(input("enter number of elements"))
mylist=[]

for i in range(n):
    num = int(input(""))
    mylist.append(num)

print(sum(mylist)/n)