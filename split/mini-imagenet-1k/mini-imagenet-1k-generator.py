import os

imagenet_dir = "/home/ec2-user/data/imagenet"

class_names = os.listdir( os.path.join(imagenet_dir, "train") )
cls_index = {c: i for (i, c) in enumerate(class_names)}

def generate(split, cls_index, train_samples_per_cls, test_samples_per_cls):
    train_path_cls = []
    test_path_cls = []
    for c in class_names:
        fns = os.listdir( os.path.join(imagenet_dir, split, c) )
        train_fns = fns[:train_samples_per_cls]
        test_fns = fns[train_samples_per_cls: train_samples_per_cls + test_samples_per_cls]

        train_path_cls += [ (os.path.join(split, c, fn), cls_index[c]) for fn in train_fns]
        test_path_cls += [ (os.path.join(split, c, fn), cls_index[c]) for fn in test_fns]

    with open("MiniImagenet-1k-train.txt", "w") as fil:
        for p, c in train_path_cls:
            fil.write("{} {}\n".format(p, c))
    
    with open("MiniImagenet-1k-test.txt", "w") as fil:
        for p, c in test_path_cls:
            fil.write("{} {}\n".format(p, c))

generate("train", cls_index, 50, 10)
