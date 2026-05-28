face = [
    ["R", "R", "G"],
    ["R", "W", "G"],
    ["B", "B", "Y"]
]

for row in face:
    for color in row:
        print(color, end=" ")
    print()

for row in face:
    for color in row:

        if color == "R":
            print("X", end=" ")

        else:
            print(color, end=" ")

    print()