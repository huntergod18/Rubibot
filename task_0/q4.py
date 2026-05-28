face = ["W", "W", "W", "White", "W", "W", "WHITE", "W", "W"]

same = True

for color in face:

    color = color.upper()

    if color == "W":
        color = "WHITE"

    if color != "WHITE":
        same = False

if same:
    print("This face is solved.")

else:
    print("This face is NOT solved.")