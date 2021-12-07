import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from rich import get_console

example = """0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2"""


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Line2d:
    p0: Point
    p1: Point

    def __repr__(self):
        return f"{self.p0.x},{self.p0.y} -> {self.p1.x},{self.p1.y}"

class Array:
    def __init__(self, shape: Tuple[int,int]):
        self.shape = shape
        self._array: List[List[int, ...]] = [[0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]

    def __setitem__(self, key, value):
        i,j = key
        self._array[i][j] = value

    def __getitem__(self, item):
        i,j = item
        return self._array[i][j]



def length(p: Point) -> float:
    return math.sqrt(p.x ** 2 + p.y ** 2)


def parse(text: str) -> List[Line2d]:
    def parse_line(t: str) -> Line2d:
        def parse_point(p: str) -> Point:
            x, y = p.split(",")
            return Point(int(x), int(y))

        start, end = t.split("->")
        return Line2d(
            *sorted([parse_point(start), parse_point(end)], key=lambda p: length(p))
        )

    lines: List[Line2d] = []
    for line in text.split("\n"):
        lines.append(parse_line(line))
    return lines


def max_dim(lines: List[Line2d]) -> Tuple[int, int]:
    points = [x for y in [[l.p0, l.p1] for l in lines] for x in y]
    max_func = lambda xy: max(points, key=lambda p: getattr(p, xy))
    return max_func("x").x + 1, max_func("y").y + 1


def fill(lines: List[Line2d]):
    array = np.zeros(max_dim(lines), dtype=int)

    for line in lines:
        # horizontal
        if line.p0.x == line.p1.x:
            xi = line.p0.x
            for yi in range(line.p0.y, line.p1.y + 1):
                array[yi, xi] += 1
        # vertical
        elif line.p0.y == line.p1.y:
            yi = line.p0.y
            for xi in range(line.p0.x, line.p1.x + 1):
                array[yi, xi] += 1

    return array


def intersections(array: np.ndarray, num_intersections: int = 2) -> np.ndarray:
    return array >= num_intersections


def num_intersections(intersection_array: np.ndarray) -> int:
    return sum(sum(intersection_array))


def pretty_print_aray(array: np.ndarray):
    console = get_console()
    def wrap(key, color): return f"[{color}]{key}[/{color}]"

    canvas = ""
    for row in array:
        for cell in row:
            if cell == 0:
                canvas += wrap('.','white')
            elif cell == 1:
                canvas += wrap(cell, 'dark_green')
            else:
                canvas += wrap(cell, 'dark_red')
        canvas += "\n"
    console.print(canvas)


lines = parse(example)
print('\n'.join([str(l) for l in lines]))
array = fill(lines)
pretty_print_aray(array)
print(num_intersections(intersections(array)))

