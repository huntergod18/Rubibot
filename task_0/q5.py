r = 0
b = 0
g = 0
w = 0
y = 0

while True:

    color = input("Enter color: ")

    color = color.upper()

    if color == "STOP":
        break

    elif color == "RED":
        r += 1

    elif color == "BLUE":
        b += 1

    elif color == "GREEN":
        g += 1

    elif color == "WHITE":
        w += 1

    elif color == "YELLOW":
        y += 1

print("R appears", r, "times" )
print("B appears", b, "times" )
print("G appears", g, "times" )
print("W appears", w, "times" )
print("Y appears", y, "times" )