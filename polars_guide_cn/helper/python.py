from collections import defaultdict


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


def search_arguments(targets, search_name):
    import inspect
    for obj in targets:
        print("-" * 20)
        for name in dir(obj):
            val = getattr(obj, name)
            try:
                args = [p.name for p in inspect.signature(val).parameters.values()]
                null_args = [arg for arg in args if search_name in arg]
                if null_args:
                    print(f'{val.__qualname__}: {",".join(null_args)}')
            except (ValueError, TypeError):
                pass


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret
