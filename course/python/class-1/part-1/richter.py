def richter():    # function definition
    scale = [1.0, 5.0, 9.1, 9.2, 9.5]
    for i in scale:
        joules = 10 ** ((1.5 * i) + 4.8)
        tnt = joules / 4.184e9
        print("%f on the Richter scale equates to %f joules and %f TNT" % (i, joules, tnt))


richter()   # calling function

usr_inpt = input("Please enter an number: ")
usr_float = float(usr_inpt)
joules = 10 ** ((1.5 * usr_float) + 4.8)
tnt = joules / 4.184e9
print("%f on the Richter scale equates to % f joules and %f TNT" % (usr_float, joules, tnt))