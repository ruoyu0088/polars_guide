def class_tree(cls):
    return { cls.__name__: [class_tree(sub_class) for sub_class in cls.__subclasses__()] }


def print_tree(p, last=True, header=''):
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    name = list(p.keys())[0]
    print(header + (elbow if last else tee) + name)
    if p[name]:
        children = p[name]
        for i, c in enumerate(children):
            print_tree(c, header=header + (blank if last else pipe), last=i == len(children) - 1)


def print_subclasses(cls):
    print_tree(class_tree(cls))
