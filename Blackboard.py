from numpy import array
from tkinter import Canvas

BLACKBOARD_SIZE = 560


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb


class Blackboard:

    def __init__(self, window, size, callback):
        self.brush_radius = 2
        self.callback = callback
        self.canvas = Canvas(window, width=BLACKBOARD_SIZE, height=BLACKBOARD_SIZE, bg="#ffffff")
        self.canvas.bind("<B1-Motion>", self.__moved)
        self.canvas.bind("<ButtonRelease-1>", callback)
        self.canvas.pack()

        self.size = size
        self.pixel_size = BLACKBOARD_SIZE // size

        self.pixels = [[0] * size for i in range(size)]

    def __moved(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if x >= self.size or y >= self.size:
            return
        self.__paint(x, y)

    def __paint(self, x, y):
        for i in range(-self.brush_radius, self.brush_radius + 1):
            for j in range(-self.brush_radius, self.brush_radius + 1):
                x1, y1 = x + i, y + j
                if not (0 <= x1 < self.size and 0 <= y1 < self.size):
                    continue
                dst = (i * i + j * j) ** 0.5
                self.__paint_pixel(x1, y1, dst)

    def __paint_pixel(self, x, y, dst):
        col = dst / (2 * self.brush_radius * self.brush_radius) ** 0.5
        col -= 0.3
        col = max(col, 0)
        col = (1 - col) ** 4

        if col > self.pixels[y][x]:
            self.pixels[y][x] = col
            col = int(255 - col * 255)
            self.__draw_rect(x, y, _from_rgb((col, col, col)))

    def __draw_rect(self, x, y, col):
        x *= self.pixel_size
        y *= self.pixel_size
        self.canvas.create_rectangle(x, y,
                                     x + self.pixel_size,
                                     y + self.pixel_size,
                                     fill=col,
                                     width=0)

    def get_array(self):
        return array(self.pixels)

    def clear(self, *args):
        self.pixels = [[0] * self.size for i in range(self.size)]
        self.canvas.create_rectangle(0, 0, BLACKBOARD_SIZE, BLACKBOARD_SIZE, fill="#ffffff", width=0)

        self.callback()

    def show(self, picture):
        for x in range(self.size):
            for y in range(self.size):
                self.pixels[y][x] = picture[y][x]
                col = int((1 - picture[y][x]) * 255)
                col = _from_rgb((col, col, col))
                self.__draw_rect(x, y, col)
        self.callback()

    def set_brush_size(self, size):
        self.brush_radius = size
