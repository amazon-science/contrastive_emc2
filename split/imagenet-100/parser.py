lines = []
with open("_ImageNet_100_test.txt", "r") as fil:
    for line in fil:
        path, clas = line.split(" ")
        path = path.split("/")
        path.pop(1)
        path = "/".join(path)
        lines.append("{} {}".format(path, clas))

with open("ImageNet_100_test.txt", "w") as fil:
    for line in lines:
        fil.write(line)