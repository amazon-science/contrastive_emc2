import os

train_dataset = {}
with open("ImageNet_100_train.txt", "r") as fil:
    for line in fil:
        img_path = line.split()[0]
        label = int(line.split()[1])
        if label in train_dataset:
            train_dataset[label].append(img_path)
        else:
            train_dataset[label] = [img_path]

for c in train_dataset:
    train_dataset[c] = train_dataset[c][:500]

with open("SubImageNet_100_train.txt", "w") as fil:
    for c in train_dataset:
        for p in train_dataset[c]:
            fil.write("{} {}\n".format(p, c))