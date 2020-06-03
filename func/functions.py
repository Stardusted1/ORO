import numpy as np
from PIL import Image as Im, ImageDraw, ImageFont
from math import ceil, floor, sqrt

import main as mn

rashmor = "rashmor"

def printToConsole(arr):
    i = 0
    while i < len(arr):
        j = 0
        while j < len(arr[i]):
            print(arr[i][j], end="")

            j += 1
        print("\n")
        i += 1


def get_center_yx_of_figure(figure: list) -> tuple:
    centr_y = len(figure) // 2
    centr_x = figure[centr_y][0][1] + int(figure[centr_y][1][1] - figure[centr_y][1][1] // 2)
    return (centr_y, centr_x)


def getByteArr(imArr) -> list:
    i = 0
    # print(imArr)
    imArrB = [[], []]
    while i < len(imArr):
        j = 0
        imArrB.insert(i, [])
        while j < len(imArr[i]):
            f1 = imArr[i][j][0]
            f2 = imArr[i][j][1]
            f3 = imArr[i][j][2]
            sumdiv = int(f1) + int(f2) + int(f3)
            sumdiv /= 3
            f = 0
            if sumdiv > 200:
                f = 1
            else:
                f = 0
            imArrB[i].insert(j, f)
            j += 1
        i += 1
    return imArrB


def getFigures(binArr) -> list:
    i = 0
    objects = []
    obj = []

    while i < len(binArr):
        j = 0
        while j < len(binArr[i]):
            if binArr[i][j] == 0:
                if binArr[i][j - 1] == 1:
                    s = j
                    while s < len(binArr[i]):
                        if binArr[i][s] == 0 and binArr[i][s + 1] == 1:
                            merge(objects, [[i, j], [i, s]])
                            j = s
                            break
                        s += 1
            j += 1
        i += 1
    return objects


def GetMaxMin_Y_X(obj) -> list:
    maxX = obj[0][1]
    minX = obj[0][1]
    maxY = max(obj)[0]
    minY = min(obj)[0]
    for x in obj:
        if x[1] >= maxX:
            maxX = x[1]
        if x[1] <= minX:
            minX = x[1]
    return [[maxY, maxX], [minY, minX]]


def get_width_height(figure):
    height = len(figure)
    width = 0
    for row in figure:
        w = row[1][1] - row[0][1]
        if w >= width:
            width = w
    return (height, width)


def GetCentrs(objects: list, baseResolution: tuple, endResolution: tuple) -> list:
    y_modifier = baseResolution[0] / endResolution[0]
    x_modifier = baseResolution[1] / endResolution[1]

    centers = []
    for x in objects:
        MaxMin = GetMaxMin_Y_X(x)

        centrX = MaxMin[1][1] + floor(((MaxMin[0][1] - MaxMin[1][1]) * 1) / 2)
        centrY = MaxMin[1][0] + floor(((MaxMin[0][0] - MaxMin[1][0]) * 1) / 2)
        centers.insert(len(centers), [centrY - 1, centrX + 1])
    return centers


def searchX(obj, binArr, startY, nextX) -> int:
    if binArr[nextX] == 0:
        obj.insert(len(obj), [startY, nextX])
        searchX(obj, binArr, startY, nextX + 1)
    else:
        return nextX


def merge(objects, point):
    y = point[0][0]
    x = point[0][1]
    y1 = point[1][0]
    x1 = point[1][1]
    i = 0
    inserted = False
    while i < len(objects):  # count obj
        j = 0
        while j < len(objects[i]):  # count points in obj
            z = 0
            while z < len(objects[i][j]):
                if objects[i][j][z][0] + mn.startPowerY >= y >= objects[i][j][z][0] - mn.startPowerY:  # if equal depth
                    if objects[i][j][z][1] + mn.powerX >= x >= objects[i][j][z][1] - mn.powerX:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][1] + mn.startPowerX >= x >= objects[i][j][z][
                    1] - mn.startPowerX:  # if equal width
                    if objects[i][j][z][0] - mn.powerY <= y <= objects[i][j][z][0] + mn.powerY:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][0] + mn.startPowerY >= y1 >= objects[i][j][z][
                    0] - mn.startPowerY:  # if equal depth
                    if objects[i][j][z][1] + mn.powerX >= x1 >= objects[i][j][z][1] - mn.powerX:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][1] + mn.startPowerX >= x1 >= objects[i][j][z][
                    1] - mn.startPowerX:  # if equal width
                    if objects[i][j][z][0] - mn.powerY <= y1 <= objects[i][j][z][0] + mn.powerY:
                        objects[i].append(point)
                        inserted = True
                        break
                z += 1
            if inserted:
                break
            j += 1
        if inserted:
            break
        i += 1
    if (not inserted):
        objects.append([point])

    if len(objects) == 0:
        objects.append([point])


def GetSqures(objects: list, baseResolution: tuple, endResolution: tuple) -> list:
    squares = []
    for figure in objects:
        fig_sq = 0
        for row in figure:
            fig_sq += row[1][1] - row[0][1] + 1
        squares.append(fig_sq)
    return squares


def getBackToImage1(imArr, centersArr, sArr):
    im = Im.open("sources/Лб_1_001L.bmp")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 8)
    c = 0

    while c < len(centersArr):
        x = centersArr[c][0]
        y = centersArr[c][1]
        draw.text((y, x), "c", (244, 65, 65), font=font)
        c += 1
    font = ImageFont.truetype("arial.ttf", 8)
    draw.text((5, mn.maxsize[0] - 8), "Кол-во -" + str(len(centersArr)), (244, 65, 65), font=font)
    draw.text((45, mn.maxsize[0] - 8), "s -" + str(min(sArr)) + "px", (244, 65, 65), font=font)
    draw.text((75, mn.maxsize[0] - 8), "S -" + str(max(sArr)) + "px", (244, 65, 65), font=font)

    im.save('my.bmp')
    im.show()


def scaleFigures(figarr, squaresArr):
    w, h = 600, 600
    scaled_im_arr = np.zeros((h, w, 4), dtype=np.uint8)

    # data for smallest figure
    min_sq = min(squaresArr)
    min_sq_ind = squaresArr.index(min_sq)
    smallest_figure = figarr[min_sq_ind]
    hw_small_figure = get_width_height(smallest_figure)
    figarr.pop(min_sq_ind)
    squaresArr.pop(min_sq_ind)

    # data for biggest figure
    max_sq = max(squaresArr)
    max_sq_ind = squaresArr.index(max_sq)
    biggest_figure = figarr[max_sq_ind]
    hw_biggest_figure = get_width_height(biggest_figure)
    figarr.pop(max_sq_ind)
    squaresArr.pop(max_sq_ind)

    # min_fig_im_arr = np.zeros((hw_small_figure[0], hw_small_figure[1], 4), dtype=np.uint8)
    # max_fig_im_arr = np.zeros((hw_biggest_figure[0], hw_biggest_figure[1], 4), dtype=np.uint8)
    # min_fig_im_arr[min_fig_im_arr == 0] = 255
    # max_fig_im_arr[max_fig_im_arr == 0] = 255
    smallest_fig_min_y_x = get_min_x_y(smallest_figure)

    biggest_fig_min_y_x = get_min_x_y(biggest_figure)

    smallest_figure_center_before = get_center_yx_of_figure(smallest_figure)

    for row in smallest_figure:
        row[0][0] -= smallest_fig_min_y_x[0]
        row[1][0] -= smallest_fig_min_y_x[0]
        row[0][1] -= smallest_fig_min_y_x[1]
        row[1][1] -= smallest_fig_min_y_x[1]

    smallest_figure = resize_to_big(smallest_figure, max_sq, min_sq)
    smallest_figure_center_after = get_center_yx_of_figure(smallest_figure)
    hw_after_resize = get_biggest_height_width(smallest_figure)
    min_fig_im_arr = np.zeros((hw_after_resize[0] + 1, hw_after_resize[1] + 1, 4), dtype=np.uint8)

    for row in smallest_figure:
        for x in range(row[0][1], row[1][1]):
            min_fig_im_arr[row[0][0]][x] = [0, 0, 0, 255]

    img = Im.fromarray(min_fig_im_arr, 'RGBA')
    img.save('small_fig.png')
    # c = Im.open('small_fig.png')
    # d = c.resize(hw_biggest_figure, 0)

    d = Im.open('small_fig.png')
    d.save('scaled_small_fig.png')
    biggest_figure_center_before = get_center_yx_of_figure(biggest_figure)
    for row in biggest_figure:
        row[0][0] -= biggest_fig_min_y_x[0]
        row[1][0] -= biggest_fig_min_y_x[0]
        row[0][1] -= biggest_fig_min_y_x[1]
        row[1][1] -= biggest_fig_min_y_x[1]

    # biggest_figure = resize_to_small(biggest_figure, max_sq, min_sq)
    biggest_figure_center_after = get_center_yx_of_figure(biggest_figure)
    hw_after_resize = get_biggest_height_width(biggest_figure)
    max_fig_im_arr = np.zeros((hw_after_resize[0] + 1, hw_after_resize[1] + 1, 4), dtype=np.uint8)

    for row in biggest_figure:
        for x in range(row[0][1], row[1][1]):
            max_fig_im_arr[row[0][0]][x] = [0, 0, 0, 255]

    img = Im.fromarray(max_fig_im_arr, 'RGBA')

    img.save('big_fig.png')
    d = Im.open('big_fig.png')
    d = d.resize(hw_small_figure)
    d.save('scaled_big_fig.png')

    """
    big_figure = figarr[max_sq_ind]
    figarr.__delitem__(max_sq_ind)
    small_figure = figarr[min_sq_ind]
    figarr.__delitem__(min_sq_ind)

    i = 0
    row = [small_figure[0]]
    ind = 0
    new_big_fig = []
    new_small_fig = []

    while i < len(small_figure):
        if small_figure[i][0] != row[ind][0]:
            # row1 = scale_to_big_row(scale_k*2, row)
            new_small_fig.insert(len(new_small_fig), row)
            row = [small_figure[i]]
            ind = 0
        elif small_figure[i] == row[ind]:
            i += 1

            continue
        else:
            row.insert(ind, small_figure[i])
            ind += 1

        i += 1

    if len(row) > 0:
        # row1 = scale_to_big_row(scale_k*2, row)
        new_small_fig.insert(len(new_small_fig), row)

    row = [big_figure[0]]
    ind = 0
    i = 0

    while i < len(big_figure):
        if big_figure[i][0] != row[ind][0]:
            # row1 = scale_to_small_row(scale_k, row)
            new_big_fig.insert(len(new_big_fig), row)
            row = [big_figure[i]]
            ind = 0
        elif big_figure[i] == row[ind]:
            i += 1
            continue
        else:
            row.insert(ind, big_figure[i])
            ind += 1

        i += 1

    if len(row) > 0:
        # row1 = scale_to_small_row(scale_k, row)
        new_big_fig.insert(len(new_big_fig), row)

    i = 0

    small_figure_height = len(new_small_fig)
    small_figure_width = 0
    small_figure_widest_row_ind = 0
    while i < len(new_small_fig):

        if len(new_small_fig[i]) > small_figure_width:
            small_figure_width = len(new_small_fig[i])
            small_figure_widest_row_ind = i
        i += 1
    i = 1
    everys = 1

    ok = not True
    ok1 = not True
    if ok:
        for row in new_small_fig:
            row.sort(key=lambda x: x[1])
        while i < small_figure_width:
            if everys < scale_k:
                x_ind = int(new_small_fig[small_figure_widest_row_ind][i][1])
                insert_row_v(new_small_fig, i, x_ind)
                everys += 1
                small_figure_width += 1
            else:
                everys = 1
            i += 1

    i = 1
    everys = 1

    if ok1:
        while i < small_figure_height:
            if everys < scale_k:
                insert_row_h(new_small_fig, i)
                everys += 1
                small_figure_height += 1
            else:
                everys = 1
            i += 1
    """

    scaled_im_arr[scaled_im_arr == 0] = 255

    scaled_im_arr[127][1] = [255, 255, 0, 1]  # y x
    f = 1
    for figure in figarr:
        n = 0

        while n < len(figure):  # rows
            for x in range(figure[n][0][1], figure[n][1][1]):
                scaled_im_arr[figure[n][0][0]][x] = [0 + f * 25, 0 + f * 35, 0 + f * 15, 255]

            n += 1
        f += 1

    # for row in new_big_fig:
    #    for point in row:
    #        scaled_im_arr[point[0]][point[1]] = [0, 0, 0, 255]
    #
    # for row in new_small_fig:
    #    for point in row:
    #        scaled_im_arr[point[0]][point[1]] = [0, 0, 0, 255]
    #
    img = Im.fromarray(scaled_im_arr, 'RGBA')
    img.save('my.png')

    img = Im.open('scaled_small_fig.png')
    img1 = Im.open('scaled_big_fig.png')
    background = Im.open('my.png')

    background.paste(img, (smallest_fig_min_y_x[1] - smallest_figure_center_after[1],
                           smallest_fig_min_y_x[0] - smallest_figure_center_after[0]), img)
    background.paste(img1, (biggest_fig_min_y_x[1] + biggest_figure_center_after[1],
                            biggest_fig_min_y_x[0] + biggest_figure_center_after[0]), img1)
    background.save('out.png')
    return {
        "figure_array": figarr,
        "biggest_figure_center_after": biggest_figure_center_after,
        "smallest_figure_center_after": smallest_figure_center_after,
        "smallest_fig_min_y_x": smallest_fig_min_y_x,
        "biggest_fig_min_y_x": biggest_fig_min_y_x
    }
    # return (figarr, biggest_figure_center_before, smallest_figure_center_before)


def get_biggest_height_width(figure: list) -> tuple:
    big_y = figure[0][0][0]
    big_x = figure[0][0][1]
    for row in figure:
        x = row[1][1]
        y = row[0][0]
        if x >= big_x:
            big_x = x
        if y >= big_y:
            big_y = y
    return big_y, big_x


def resize_to_big(figure: list, sq_big, sq_small) -> list:
    # size - height, width
    delta_width = ceil(sqrt(sq_big) / sqrt(sq_small))
    delta_width += sqrt(delta_width)
    delta_heigth = sqrt(sq_big) / sqrt(sq_small)
    delta_heigth += sqrt(delta_heigth)
    ind = int(delta_heigth)
    i = 1
    small_figure = []
    n = 0
    while n < len(figure):
        row = figure[n]
        row1 = list()
        row1.append([row[0][0], row[0][1]])
        row1.append([row[1][0], row[1][1]])
        while i != ind:
            small_figure.append([[row1[0][0], row1[0][1]], [row1[1][0], row1[1][1]]])
            row1[0][0] += 1
            row1[1][0] += 1
            i += 1
        f = n
        while n < len(figure):
            figure[n][0][0] += ind - 2
            figure[n][1][0] += ind - 2
            n += 1
        n = f
        i = 1
        n += 1

    for row in small_figure:
        row[0][1] *= int(delta_heigth // 2) + 3
        row[1][1] *= int(delta_width) - 1

    '''
        try:
        new_len = int(len(figure)*(delta_heigth))
        while i < new_len:
            if ind == 1:
                row[0][0] += 1
                row[1][0] += 1
                figure.insert(i, row)
                ind = delta_heigth
            elif ind < delta_heigth:
                row[0][0] += 1
                row[1][0] += 1
                figure.insert(i, [[row[0][0], row[0][1]], [row[1][0], row[1][1]]])
                ind -= 1
            elif ind == delta_heigth:
                row = list()
                row.append([int(figure[i][0][0]), int(figure[i][0][1])])
                row.append([(figure[i][1][0]), (figure[i][1][1])])
                ind -= 1
            i += 1
        lne = len(figure)
    '''
    return small_figure


def resize_to_small(figure: list, sq_big: int, sq_small: int) -> list:
    delta_width = ceil(sqrt(sq_big) / sqrt(sq_small))
    delta_width += int(sqrt(delta_width))
    delta_heigth = ceil(sqrt(sq_big) / sqrt(sq_small))
    delta_heigth += int(sqrt(delta_heigth))
    ind = int(delta_heigth)
    n = 0

    while n < len(figure):
        ind = int(delta_heigth)
        while ind > 0:
            try:
                figure.pop(n + 1)
            except:
                break
            ind -= 1
        n += 1

    f = 0
    while f < len(figure):
        figure[f][0][0] = f
        figure[f][1][0] = f
        f += 1

    for row in figure:
        delta = row[0][1] - row[1][1]
        delta_1 = delta // 2
        delta_1 = delta - (delta_1 / delta_heigth)
        row[0][1] += int(delta_1)
        row[1][1] -= int(delta_1)

    return figure


def get_left_upper_position(figure):
    h = (len(figure)) // 2
    d_w = (figure[h][1][1] - figure[h][0][1]) // 2
    w = (figure[h][0][1] + d_w)
    return (figure[h][0][0], w)


def get_min_x_y(figure):
    min_x = figure[0][0][1]
    min_y = figure[0][0][0]
    for row in figure:
        x = row[0][1]
        if x < min_x:
            min_x = x
    return (min_y, min_x)


def swap_figures(sc_dict: dict):
    fugure_array = sc_dict["figure_array"]
    biggest_figure_center_after = sc_dict["biggest_figure_center_after"]
    smallest_figure_center_after = sc_dict["smallest_figure_center_after"]
    biggest_fig_min_y_x = sc_dict["biggest_fig_min_y_x"]
    smallest_fig_min_y_x = sc_dict["smallest_fig_min_y_x"]

    # for small figure
    # smallest_fig_min_y_x[1] - smallest_figure_center_after[1],
    # smallest_fig_min_y_x[0] - smallest_figure_center_after[0]

    # for big figure
    # biggest_fig_min_y_x[1] + biggest_figure_center_after[1],
    # biggest_fig_min_y_x[0] + biggest_figure_center_after[0]

    img = Im.open('scaled_small_fig.png')
    img1 = Im.open('scaled_big_fig.png')
    background = Im.open('my.png')

    background.paste(img1, (smallest_fig_min_y_x[1],
                            smallest_fig_min_y_x[0]), img1)

    background.paste(img, (biggest_fig_min_y_x[1] + smallest_figure_center_after[1] // 2,
                           biggest_fig_min_y_x[0]), img)

    background.save('place_change.png')
