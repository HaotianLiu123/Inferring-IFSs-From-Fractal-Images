import json
import random
import os.path

from fractal import Point, exec_command

from collections import OrderedDict

# String formatting functions
Point.__str__ = lambda self: "x".join(map(str, self))
basic_str = lambda el: str(el).lstrip("(").rstrip(")")  # Complex or model name
item_str = lambda k, v: (basic_str(v) if k in ["model", "c"] else
                         "{}={}".format(k, v))
filename = lambda kwargs: "_".join(item_str(*pair) for pair in kwargs.items())
random.seed(42)
kwargs_list = []
for i in range(10000):
    real_part = random.uniform(-1, 1)
    image_part = random.uniform(-1, 1)
    depth = random.randint(10, 30)
    zoom = random.uniform(0.5, 1.5)
    temp = OrderedDict([("model", "julia"),
                        ("c", complex(real_part, image_part)),
                        ("size", Point(512, 512)),
                        ("depth", depth),
                        ("zoom", zoom), ])
    kwargs_list.append(temp)
if __name__ == "__main__":
    count = 0
    file_name_list = []
    for kwargs in kwargs_list:
        temp_dict = {}
        temp_dict['file_name'] = filename(kwargs)
        file_name_list.append(temp_dict)
        kwargs["output"] = "Data/{}.png".format(filename(kwargs))
        if not os.path.exists('Data'):
            os.makedirs('Data')
        exec_command(kwargs)
        count += 1
        print(count)
