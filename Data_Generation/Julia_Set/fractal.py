#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on Tue Apr 08 08:45:59 2014
# License is MIT, see COPYING.txt for more details.
# @author: Danilo de Jesus da Silva Bellini
"""
Julia and Mandelbrot fractals image creation
"""

from __future__ import division, print_function
import pylab, argparse, collections, inspect, functools
from itertools import takewhile
import time
import multiprocessing

# 定义了Point元组，这个元组里面有两个值分别是x和y
Point = collections.namedtuple("Point", ["x", "y"])

def pair_reader(dtype):
  return lambda data: Point(*map(dtype, data.lower().split("x")))


DEFAULT_SIZE = "512x512"
DEFAULT_DEPTH = "256"
DEFAULT_ZOOM = "1"
DEFAULT_CENTER = "0x0"
DEFAULT_COLORMAP = "cubehelix"


def repeater(f):
  """
  Returns a generator function that returns a repeated function composition
  iterator (generator) for the function given, i.e., for a function input
  ``f`` with one parameter ``n``, calling ``repeater(f)(n)`` yields the
  values (one at a time)::

     n, f(n), f(f(n)), f(f(f(n))), ...

  Examples
  --------

  >>> func = repeater(lambda x: x ** 2 - 1)
  >>> func
  <function ...>
  >>> gen = func(3)
  >>> gen
  <generator object ...>
  >>> next(gen)
  3
  >>> next(gen) # 3 ** 2 - 1
  8
  >>> next(gen) # 8 ** 2 - 1
  63
  >>> next(gen) # 63 ** 2 - 1
  3968

  """
  # 这个函数能够接受一个参数f，并返回一个新的函数，这个生成器函数可以产生一个无限序列，这个序列包含对函数f的重复组合
  @functools.wraps(f)
  def wrapper(n):
    val = n
    while True:
      yield val
      val = f(val)
  return wrapper


def amount(gen, limit=float("inf")):
  """
  Iterates through ``gen`` returning the amount of elements in it. The
  iteration stops after at least ``limit`` elements had been iterated.

  Examples
  --------
  # 函数接受了一个迭代器gen和一个可选的参数limit，函数的目标是迭代gen中的元素，并返回实际迭代的元素数量，但是在达到指定的limit后停止迭代
  >>> amount(x for x in "abc")
  3
  >>> amount((x for x in "abc"), 2)
  2
  >>> from itertools import count
  >>> amount(count(), 5) # Endless, always return ceil(limit)
  5
  >>> amount(count(start=3, step=19), 18.2)
  19
  """
  size = 0
  for unused in gen:
    size += 1
    if size >= limit:
      break
  return size


def in_circle(radius):
  # 这个函数接受的是一个半径的长度，返回的是一个lambda的函数，这个lambda函数能够接受一个复数z作为参数，然后计算并返回布尔值，判断复数z是否在以原点为中心，半径为radius的圆内
  """ Returns ``abs(z) < radius`` boolean value function for a given ``z`` """
  return lambda z: z.real ** 2 + z.imag ** 2 < radius ** 2


def fractal_eta(z, func, limit, radius=2):
  """
  Fractal Escape Time Algorithm for pixel (x, y) at z = ``x + y * 1j``.
  Returns the fractal value up to a ``limit`` iteration depth.
  """
  # 这是一个分形逃逸算法，这个算法用于生成分形图像，用于确定复平面上每个点在迭代过程中是否逃逸出某个范围的算法
  return amount(takewhile(in_circle(radius), repeater(func)(z)), limit)


def cqp(c):
  """ Complex quadratic polynomial, function used for Mandelbrot fractal """
  # 这个就是mandelbort fractal
  # cqp(c)返回的lambda 函数接受一个复数参数z，然后使用复二次多项式的公式计算结果，就是z^2+c
  # 这里的z相当于迭代的开始起点能够反应的是这个点所在的复平面区域的性质。而c是你要求的常熟，可能会给整个图形带来不一样的形状
  return lambda z: z ** 2 + c

# def fractal_eta(z, func, limit, radius=2):
#   """
#   Fractal Escape Time Algorithm for pixel (x, y) at z = ``x + y * 1j``.
#   Returns the fractal value up to a ``limit`` iteration depth.
#   """
#   # 这是一个分形逃逸算法，这个算法用于生成分形图像，用于确定复平面上每个点在迭代过程中是否逃逸出某个范围的算法
#   return amount(takewhile(in_circle(radius), repeater(func)(z)), limit)
#
#

def get_model(model, depth, c):
  """
  Returns the fractal model function for a single pixel.
  """
  # 所以对于julia集来说，这里面的参数包含了c，（x,y），depth
  # 对于Mandelbrot集来说，这里面的参数包含了(x,y)和depth，这里的c就是常数0
  if model == "julia":
    func = cqp(c)
    # 这里的function里面的c是一个固定的值，但是起点是提前给定的point对应的x和y
    return lambda x, y: fractal_eta(x + y * 1j, func, depth)
  if model == "mandelbrot":
    #   这里的z是保持不变的，一直都是0，变得是后面的常数项c，这里的c就是给定的起始点
    return lambda x, y: fractal_eta(0, cqp(x + y * 1j), depth)
  raise ValueError("Fractal not found")


def generate_fractal(model, c=None, size=pair_reader(int)(DEFAULT_SIZE),
                     depth=int(DEFAULT_DEPTH), zoom=float(DEFAULT_ZOOM),
                     center=pair_reader(float)(DEFAULT_CENTER)):
  """
  2D Numpy Array with the fractal value for each pixel coordinate.
  """
  num_procs = multiprocessing.cpu_count()
  print('CPU Count:', num_procs)
  start = time.time()

  # Create a pool of workers, one for each row
  pool = multiprocessing.Pool(num_procs)
  procs = [pool.apply_async(generate_row,
                            [model, c, size, depth, zoom, center, row])
           for row in range(size[1])]

  # Generates the intensities for each pixel
  img = pylab.array([row_proc.get() for row_proc in procs])

  print('Time taken:', time.time() - start)
  return img


def generate_row(model, c, size, depth, zoom, center, row):
  """
  Generate a single row of fractal values, enabling shared workload.
  """
  # 这里的size是最终生成的图像的宽度和高度，这里的depth是迭代的深度和迭代的次数，这里的zoom是生成的缩放因子，表示缩放级别
  func = get_model(model, depth, c)
  width, height = size
  cx, cy = center
  side = max(width, height)
  sidem1 = side - 1
  deltax = (side - width) / 2 # Centralize
  deltay = (side - height) / 2
  y = (2 * (height - row + deltay) / sidem1 - 1) / zoom + cy
  return [func((2 * (col + deltax) / sidem1 - 1) / zoom + cx, y)
          for col in range(width)]


def img2output(img, cmap=DEFAULT_COLORMAP, output=None, show=False):
  """ Plots and saves the desired fractal raster image """
  if output:
    pylab.imsave(output, img, cmap=cmap)
  if show:
    pylab.imshow(img, cmap=cmap)
    pylab.show()


def call_kw(func, kwargs):
  """ Call func(**kwargs) but remove the possible unused extra keys before """
  keys = inspect.getargspec(func).args
  kwfiltered = dict((k, v) for k, v in kwargs.items() if k in keys)
  return func(**kwfiltered)


def exec_command(kwargs):
  """ Fractal command from a dictionary of keyword arguments (from CLI) """
  kwargs["img"] = call_kw(generate_fractal, kwargs)
  call_kw(img2output, kwargs)


def cli_parse_args(args=None, namespace=None):
  """
  CLI (Command Line Interface) parsing based on ``ArgumentParser.parse_args``
  from the ``argparse`` module.
  """
  # CLI interface description
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="by Danilo J. S. Bellini",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("model", choices=["julia", "mandelbrot"],
                      help="Fractal type/model")
  parser.add_argument("c", nargs="*", default=argparse.SUPPRESS,
                      help="Single Julia fractal complex-valued constant "
                           "parameter (needed for julia, shouldn't appear "
                           "for mandelbrot), e.g. -.7102 + .2698j (with the "
                           "spaces), or perhaps with zeros and 'i' like "
                           "-0.6 + 0.4i. If the argument parser gives "
                           "any trouble, just add spaces between the numbers "
                           "and their signals, like '- 0.6 + 0.4 j'")
  parser.add_argument("-s", "--size", default=DEFAULT_SIZE,
                      type=pair_reader(int),
                      help="Size in pixels for the output file")
  parser.add_argument("-d", "--depth", default=DEFAULT_DEPTH,
                      type=int,
                      help="Iteration depth, the step count limit")
  parser.add_argument("-z", "--zoom", default=DEFAULT_ZOOM,
                      type=float,
                      help="Zoom factor, assuming data is shown in the "
                           "[-1/zoom; 1/zoom] range for both dimensions, "
                           "besides the central point displacement")
  parser.add_argument("-c", "--center", default=DEFAULT_CENTER,
                      type=pair_reader(float),
                      help="Central point in the image")
  parser.add_argument("-m", "--cmap", default=DEFAULT_COLORMAP,
                      help="Matplotlib colormap name to be used")
  parser.add_argument("-o", "--output", default=argparse.SUPPRESS,
                      help="Output to a file, with the chosen extension, "
                           "e.g. fractal.png")
  parser.add_argument("--show", default=argparse.SUPPRESS,
                      action="store_true",
                      help="Shows the plot in the default Matplotlib backend")

  # Process arguments
  ns_parsed = parser.parse_args(args=args, namespace=namespace)
  if ns_parsed.model == "julia" and "c" not in ns_parsed:
    parser.error("Missing Julia constant")
  if ns_parsed.model == "mandelbrot" and "c" in ns_parsed:
    parser.error("Mandelbrot has no constant")
  if "output" not in ns_parsed and "show" not in ns_parsed:
    parser.error("Nothing to be done (no output file name nor --show)")
  if "c" in ns_parsed:
    try:
      ns_parsed.c = complex("".join(ns_parsed.c).replace("i", "j"))
    except ValueError as exc:
      parser.error(exc)

  return vars(ns_parsed)

if __name__ == "__main__":
  exec_command(cli_parse_args())
