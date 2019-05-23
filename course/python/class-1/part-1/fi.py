netAmount = 0
while True:
    s = input("enter the operation and then amount")
    if not s:
        break
    values = s.split()
    operation = values[0]
    amount = int(values[1])
    if operation == "D":
        netAmount += amount
    elif operation == "W":
        netAmount -= amount
    else:
        pass
    print(netAmount)
