# python program which will accept two numerical values and find biggest one of them by using anonymous functins


findmax=lambda a,b: a if a>b else b



#main program
a,b=float(input("Enter first value:")),float(input("enter second value:"))
res=findmax(a,b)
print("Big({},{})={}".format(a,b,res))


