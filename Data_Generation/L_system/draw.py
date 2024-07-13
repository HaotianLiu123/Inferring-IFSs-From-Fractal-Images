import json
import os.path
import pylab as pl
from matplotlib import collections
import math

def get_lines(rule, n, angle):
    """
        根据规则生成边信息
        rule: 生成规则

        返回值：每次迭代所得的X坐标数组, Y坐标数组
    """
    # 初始化
    info = rule['S']
    # 按rule中的定义展开构造规则
    for i in range(n):
        temp_info = []
        for c in info:
            if c in rule:
                temp_info.append(rule[c])
            else:
                temp_info.append(c)
        info = "".join(temp_info)
    # 这里保存的是direction，下面保存的是angle，也就是画笔的运行方向和运行的角度。
    a = angle
    d = 0
    p = (0.0, 0.0)  # 初始坐标
    l = 1  # 步长
    lines = []
    stack = []

    # 生成这里面的info的信息
    # print(info)
    # 开始生成边信息
    for c in info:
        # 绘制一条边
        if c in "Ff":
            r = (d) % 360 * math.pi / 180
            t = p[0] + l * math.cos(r), p[1] + l * math.sin(r)
            lines.append(((p[0], p[1]), (t[0], t[1])))
            p = t
        # 旋转
        elif c == "+":
            d += a
        # 旋转
        elif c == "-":
            d -= a
        elif c == "[":
            stack.append((p, d))
        elif c == "]":
            p, d = stack[-1]
            del stack[-1]
        elif c == '!':
            a = a * (-1)
        elif c == '|':
            d = d + 180
    return lines


def draw_image(rule, n, angle):
    lines = get_lines(rule, n, angle)
    linecollections = collections.LineCollection(lines, colors='white', linewidths=0.218)
    pl.axes().add_collection(linecollections, autolim=True)
    pl.axis("equal")
    pl.xticks([])
    pl.yticks([])
    pl.gca().xaxis.set_major_locator(pl.NullLocator())
    pl.gca().yaxis.set_major_locator(pl.NullLocator())
    pl.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)


name_list=['re-coil', 'the-triffids', 'tree-horse', 'ordered-chaos', 'try-force', 'metamorphosis', 'pulse-engine', 'pollenate', 'the-park-at-night', 'leviathon', 'the-DNA-dance', 'snake-charmer', 'bubble-trouble', 'wormly', 'platterstar', 'heisenberg', 'trigon', 'spirocopter', 'johnny-lee', 'angel-flight', 'stargaze', 'mortal-coil', 'house-rules', 'jamiroquai---rippletwist', 'coalescence', 'battling-dragon-tails', 'snapdragon', 'dragon-spirograph', 'gemini', 'samurai', 'dragonfish', '3d-pyramids', 'the-bomb', 'ornate-clock', 'manta-ray', 'twistmas-tree', 'clover', 'trifecta', 'brainchild', 'amoeba', 'clockwork', 'helicopter-plant-(drag-left)', 'coral', 'the-slug', 'tornado', 'politic', 'spin-engine', 'rorschach-test', 'dance-for-me', 'flagella', 'spindlethrift', 'gears', 'wigglesaur', '3d-tusk---gearspin', 'crystal-orchid', 'slamurai', 'fire!', 'single-3d-blade', 'flyweight']

for name in name_list:
    rule_path = 'Data_json/'+ name +'.json'
    save_path = 'Data/'
    print(name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(rule_path, 'r', encoding='utf-8') as f:
        rule = json.load(f)[0]
    for n in range(rule['iteration'], rule['iteration']+12, 2):
        for angle in range(5, 180, 5):
            pl.figure(figsize=(5, 5))
            draw_image(rule, n, angle)
            pl.gca().set_facecolor("black")
            pl.savefig(save_path+"/"+name+ "_angle_{}_iter_{}.png".format(angle,n), dpi=1024)
            print("Done!")

