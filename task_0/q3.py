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

print("R ", r )
print("B ", b )
print("G ", g )
print("W ", w )
print("Y ", y )