stack = [3, 5, 4]
print(stack)
# adding number to stack(performing push information)
c=1
while c!=0 :
    print("enter 1 for append 2 for pop 3 to empty the stack 4 to sort and 0 to exit")
    c = int(input("enter operation"))
    if c==1:
        stack.append(6)
        print(stack)
    elif c==2:
        stack.pop()
        print(stack)
    elif c==3:
        stack.clear()
        print(stack)
    elif c==4:
        stack.sort()
        print(stack)
    else :
        print("incorrect selection")