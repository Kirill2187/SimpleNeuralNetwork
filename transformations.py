import numpy as np
from random import randint


def __shift(picture, dx, dy):
    return np.roll(np.roll(picture, dx), -dy, axis=0)


def shift_right(picture):
    cnt = 0
    arr = np.any(picture, axis=0)
    for x in range(-1, -10, -1):
        if arr[x]:
            break
        cnt += 1
    return __shift(picture, randint(min(cnt, 1), cnt), 0)


def shift_left(picture):
    cnt = 0
    arr = np.any(picture, axis=0)
    for x in range(0, 10):
        if arr[x]:
            break
        cnt += 1
    return __shift(picture, -randint(min(cnt, 1), cnt), 0)


def shift_up(picture):
    cnt = 0
    arr = np.any(picture, axis=1)
    for x in range(0, 10):
        if arr[x]:
            break
        cnt += 1
    return __shift(picture, 0, randint(min(cnt, 1), cnt))


def shift_down(picture):
    cnt = 0
    arr = np.any(picture, axis=1)
    for x in range(-1, -10, -1):
        if arr[x]:
            break
        cnt += 1
    return __shift(picture, 0, -randint(min(cnt, 1), cnt))
