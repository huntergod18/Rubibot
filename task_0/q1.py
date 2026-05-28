face = ["red", "blue", "Red", "white", "RED", "blue", "White", "green", "Green"]
r , b , w , g = 0 , 0 , 0 , 0

for color in face:

    color = color.upper()
    if color == "RED":
        r += 1

    elif color == "BLUE":
        b += 1

    elif color == "WHITE":
        w += 1

    elif color == "GREEN":
        g += 1

print("R appears", r, "times")
print("B appears", b, "times")
print("W appears", w, "times")
print("G appears", g, "times")