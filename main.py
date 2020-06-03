from enum import Enum

import numpy as np
from PIL import Image as Im, ImageDraw, ImageFont
from math import ceil, sqrt

im_path_Lab_1 = "sources/Лб_1_002.jpg"  ##здесь указать изображение
im_pathBMP = im_path_Lab_1 + ".bmp"  ## 1,2,5,6,8,10,11,13,20,26,28,30,31,32

im_path_Lab_2 = "sources/Лб_4_032.jpg"
im_pathBMP_Lab_2 = im_path_Lab_2 + ".bmp"

koeficient = 0.925 # 0.925
approx_value = 0
magic_number = 90
square_color = (241, 244, 66)

powerX, powerY, startPowerX, startPowerY = 4, 4, 4, 4

powerXL = 10
powerYL = 10
startPowerXL = 10
startPowerYL = 10

Im.open(im_path_Lab_1).save(im_pathBMP)
fp = im_pathBMP
baseResolution = (600, 600)
maxsize = (600, 600)


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


def getByteArr(imArr) -> (list, int):
    i = 0
    # print(imArr)
    max_byte = 0
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
    return imArrB, max_byte


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
                        if binArr[i][s-1] == 0 and binArr[i][s] == 1:
                            merge(objects, [[i, j], [i, s]])
                            j = s
                            break
                        s += 1
            j += 1
        i += 1
    return objects


def getFigures1(binArr) -> list:
    i = 0
    objects = []

    while i < len(binArr):
        j = 0
        while j < len(binArr[i]):
            if binArr[i][j] == 0:
                if binArr[i][j - 1] == 1:
                    s = j
                    while s < len(binArr[i]):
                        if binArr[i][s - 1] == 0 and binArr[i][s] == 1:
                            merge(objects, [[i, j], [i, s]])
                            j = s
                            break
                        s += 1
            j += 1
        i += 1
    return objects




def GetMaxMin_Y_X(fig) -> list:
    min_x_y = get_min_x_y(fig)
    max_x_y = get_max_x_y(fig)
    return [max_x_y, min_x_y]


def get_width_height(figure):
    height = len(figure)
    width = 0
    for row in figure:
        w = row[1][1] - row[0][1]
        if w >= width:
            width = w
    return (height, width)


def GetCentrs_Y_X(objects: list) -> list:
    centers = []
    for obj in objects:
        max_min = GetMaxMin_Y_X(obj)
        center_x = ((max_min[0][1] - max_min[1][1]) // 2) + max_min[1][1]
        center_y = ((max_min[0][0] - max_min[1][0]) // 2) + max_min[1][0]
        centers.append([center_y, center_x])
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
    i = len(objects) - 1
    inserted = False
    while i > -1:  # count obj
        j = (len(objects[i]) - 1)
        while j > -1:  # count points in obj
            z = 0
            while z < len(objects[i][j]):
                if objects[i][j][z][0] + startPowerY >= y >= objects[i][j][z][0] - startPowerY:  # if equal depth
                    if objects[i][j][z][1] + powerX >= x >= objects[i][j][z][1] - powerX:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][1] + startPowerX >= x >= objects[i][j][z][1] - startPowerX:  # if equal width
                    if objects[i][j][z][0] - powerY <= y <= objects[i][j][z][0] + powerY:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][0] + startPowerY >= y1 >= objects[i][j][z][0] - startPowerY:  # if equal depth
                    if objects[i][j][z][1] + powerX >= x1 >= objects[i][j][z][1] - powerX:
                        objects[i].append(point)
                        inserted = True
                        break
                elif objects[i][j][z][1] + startPowerX >= x1 >= objects[i][j][z][1] - startPowerX:  # if equal width
                    if objects[i][j][z][0] - powerY <= y1 <= objects[i][j][z][0] + powerY:
                        objects[i].append(point)
                        inserted = True
                        break
                z += 1
            if inserted:
                break
            j -= 1
        if inserted:
            break
        i -= 1
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
    im = Im.open(im_path_Lab_1)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 16)
    c = 0

    while c < len(centersArr):
        x = centersArr[c][0]
        y = centersArr[c][1]
        draw.text((y, x), "c", (244, 65, 65), font=font)
        c += 1
    font = ImageFont.truetype("arial.ttf", 15)
    draw.text((5, maxsize[0] - 15), "Кол-во -" + str(len(centersArr)), (244, 65, 65), font=font)
    draw.text((100, maxsize[0] - 15), "s -" + str(min(sArr)) + "px", (244, 65, 65), font=font)
    draw.text((200, maxsize[0] - 15), "S -" + str(max(sArr)) + "px", (244, 65, 65), font=font)

    im.save('lab1.bmp')


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
    background.save('Lab2.png')
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


def get_max_x_y(figure):
    max_x = figure[0][0][1]
    max_y = figure[-1][0][0]
    for row in figure:
        x = row[1][1]
        if x > max_x:
            max_x = x
    return (max_y, max_x)


def swap_figures(sc_dict: dict):
    figure_array = sc_dict["figure_array"]
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

    background.save('Lab2_place_change.png')
    return {
        "figure_array": figure_array,
        "biggest_figure_center_after": biggest_figure_center_after,
        "smallest_figure_center_after": smallest_figure_center_after,
        "smallest_fig_min_y_x": [biggest_fig_min_y_x[0], biggest_fig_min_y_x[1]
        + smallest_figure_center_after[1] // 2],
        "biggest_fig_min_y_x": smallest_fig_min_y_x
    }


def rotate_left(sc_dict):
    smallest_figure_center_after = sc_dict["smallest_figure_center_after"]
    biggest_fig_min_y_x = sc_dict["biggest_fig_min_y_x"]
    smallest_fig_min_y_x = sc_dict["smallest_fig_min_y_x"]
    new_big_fig_pos = [smallest_fig_min_y_x[0] - biggest_fig_min_y_x[0],
                       smallest_fig_min_y_x[1] - biggest_fig_min_y_x[1]]

    t = new_big_fig_pos[0]
    new_big_fig_pos[0] = new_big_fig_pos[1] * -1  # y
    new_big_fig_pos[1] = t  # x

    new_big_fig_pos[0] += biggest_fig_min_y_x[0]
    new_big_fig_pos[1] += biggest_fig_min_y_x[1]

    sc_small_im = Im.open('scaled_small_fig.png')
    sc_big_im = Im.open('scaled_big_fig.png')
    background = Im.open('my.png')

    background.paste(sc_small_im, (new_big_fig_pos[1],
                                   new_big_fig_pos[0] - round(smallest_figure_center_after[0] * 1.75)), sc_small_im)
    background.paste(sc_big_im, (biggest_fig_min_y_x[1],
                                 biggest_fig_min_y_x[0]), sc_big_im)
    background.save('Lab3_90L.png')


def rotate_right(sc_dict):
    smallest_figure_center_after = sc_dict["smallest_figure_center_after"]
    biggest_fig_min_y_x = sc_dict["biggest_fig_min_y_x"]
    smallest_fig_min_y_x = sc_dict["smallest_fig_min_y_x"]

    # new_coord_center = (0, 0)

    new_big_fig_pos = [smallest_fig_min_y_x[0] - biggest_fig_min_y_x[0],
                       smallest_fig_min_y_x[1] - biggest_fig_min_y_x[1]]

    t = new_big_fig_pos[1]
    new_big_fig_pos[1] = new_big_fig_pos[0] * -1  # x
    new_big_fig_pos[0] = t  # y

    new_big_fig_pos[0] += biggest_fig_min_y_x[0]
    new_big_fig_pos[1] += biggest_fig_min_y_x[1]

    sc_small_im = Im.open('scaled_small_fig.png')
    sc_big_im = Im.open('scaled_big_fig.png')
    background = Im.open('my.png')

    background.paste(sc_small_im, (new_big_fig_pos[1] - round(smallest_figure_center_after[1] * 1.75),
                                   new_big_fig_pos[0]), sc_small_im)
    background.paste(sc_big_im, (biggest_fig_min_y_x[1],
                                 biggest_fig_min_y_x[0]), sc_big_im)
    background.save('Lab3_90R.png')


def get_rgbsum_image_arr(imArr):
    byte_arr = []
    max_rgb_arr = []
    for row in imArr:
        byte_row = []
        max_rgb = 0
        for point in row:
            f1 = point[0]
            f2 = point[1]
            f3 = point[2]
            p = 0
            p = (p + f1 + f2 + f3) / 3
            byte_row.append(p)
            if max_rgb < p:
                max_rgb = p
        byte_arr.append(byte_row)
        max_rgb_arr.append(max_rgb)
    return byte_arr, max_rgb_arr


def get_binary_array(imArr: list, max_b_arr: list):
    i = 0
    pixel_sum = 0
    bin_arr = []
    while i < len(imArr):
        j = 0
        row = []
        while j < len(imArr[i]):
            point = 0
            if pixel_sum >= koeficient * max_b_arr[i]:
                point = 1
                pixel_sum -= koeficient * max_b_arr[i]
            elif pixel_sum < koeficient * max_b_arr[i]:
                point = 0
                pixel_sum -= magic_number  # WTF ????????????????????
            pixel_sum += imArr[i][j]
            row.append(point)
            j += 1
        bin_arr.append(row)
        i += 1
    return bin_arr


def reform_to_image(bin_arr, height_width):
    h = height_width[0]
    w = height_width[1]
    scaled_im_arr = np.zeros((h, w, 3), dtype=np.uint8)
    i = 0
    while i < len(bin_arr):
        j = 0
        while j < len(bin_arr[i]):
            scaled_im_arr[i][j] = bin_arr[i][j]
            j += 1
        i += 1
    scaled_im_arr[scaled_im_arr == 1] = 255
    img = Im.fromarray(scaled_im_arr)
    img.save("Lab_4_impulse.png")


def merge_impulse(objects, point):
    y = point[0]
    x = point[1]
    i = 0
    inserted = False
    while i < len(objects):  # count obj
        j = 0
        while j < len(objects[i]):  # count points in obj
            if objects[i][j][0] + startPowerY >= y >= objects[i][j][0] - startPowerY:  # if equal depth
                if objects[i][j][1] + powerX >= x >= objects[i][j][1] - powerX:
                    objects[i].append(point)
                    inserted = True
                    break
            elif objects[i][j][1] + startPowerX >= x >= objects[i][j][1] - startPowerX:  # if equal width
                if objects[i][j][0] - powerY <= y <= objects[i][j][0] + powerY:
                    objects[i].append(point)
                    inserted = True
                    break
            if inserted:
                break
            j += 1
        if inserted:
            break
        i += 1

    if not inserted:
        objects.append([point])

    if len(objects) == 0:
        objects.append([point])


def getFiguresInImpulseImage(bin_arr):
    objects = []
    i = 0
    while i < len(bin_arr):
        j = 0
        while j < len(bin_arr[i]):
            if bin_arr[i][j] == 0:
                merge_impulse(objects, [i, j])
            j += 1
        i += 1
    return objects


def delete_trash(figure_array):
    max_len = 0
    for fig in figure_array:
        if len(fig) > max_len:
            max_len = len(fig)

    i = 0
    while i < len(figure_array):
        if len(figure_array[i]) < max_len // 1.5:
            figure_array.pop(i)
            continue
        i += 1
    return figure_array


def get_w_h(figure_array):
    w_h_array = []
    for figure in figure_array:
        max_x = figure[0][1]  # y = point[0]
        min_x = figure[0][1]  # x = point[1]
        max_y = figure[-1][0]
        min_y = figure[0][0]
        for point in figure:
            if point[1] > max_x:
                max_x = point[1]
            if point[1] < min_y:
                min_x = point[1]
        fig = Figure()
        fig.min_y = min_y - approx_value
        fig.max_y = max_y + approx_value
        fig.min_x = min_x - approx_value
        fig.max_x = max_x + approx_value
        w_h_array.append(fig)
    return w_h_array


class Figure:
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0

    def width(self):
        return self.max_x - self.min_x

    def height(self):
        return self.max_y - self.min_y

    def square(self):
        return self.width() * self.height()

    def center_y(self):
        return (abs((self.max_y - self.min_y)) // 2) + self.min_y

    def center_x(self):
        return abs(((self.max_x - self.min_x)) // 2) + self.min_x


def draw_to_image(width_heigt_arr, imArrS):
    imArr = []
    imArr = imArrS.copy()
    for fig in width_heigt_arr:
        i = fig.max_y
        for x in range(fig.min_x, fig.max_x):
            imArr[i][x] = square_color
        i = fig.min_y
        for x in range(fig.min_x, fig.max_x):
            imArr[i][x] = square_color
        while i < fig.max_y:
            j = fig.min_x
            j1 = fig.max_x
            imArr[i][j] = square_color
            imArr[i][j1] = square_color
            i += 1
    img = Im.fromarray(imArr)
    img.save("Lab_5_impulse.png")
    for fig in width_heigt_arr:
        im = Im.open("Lab_5_impulse.png")
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("arial.ttf", 12)
        # draw.text((83, 56), "c", (244, 65, 65), font=font)
        draw.text((fig.center_x() - 5, fig.center_y() - 5), "c", (244, 65, 65), font=font)
        font = ImageFont.truetype("arial.ttf", 9)
        square = fig.square()
        draw.text((fig.min_x, fig.max_y), "S - " + str(square) + "px", (244, 65, 65), font=font)
    im.save('Lab_5_impulse.png')


def main():
    # im = Im.open(im_path_Lab_2)
    # # tn_image = Im.Image.thumbnail(im, maxsize, Im.ANTIALIAS)
    # # im.save("sources/Лб_1_001L.bmp")
    # print("\r10%", end="")
    #
    # imArr = np.asarray(im)
    # print("\r20%", end="")
    # res = get_rgbsum_image_arr(imArr)
    # tup, max_bytes_arr = res[0], res[1]
    # #tup = getByteArr(imArr)
    # bin_arr = get_binary_array(tup, max_bytes_arr)
    # imArrB = tup[0]
    # max_byte = tup[1]
    # # printToConsole(imArrB)
    # print("\r30%", end="")
    #
    # figarr = getFigures(imArrB)
    # # printToConsole(figarr)
    # print("\r40%", end="")
    #
    # squares = GetSqures(figarr, baseResolution, maxsize)
    # # print(squares)
    # print("\r50%", end="")
    #
    # centers = GetCentrs_Y_X(figarr)
    # # print(centers)
    # print("\r60%", end="")
    #
    # getBackToImage1(imArr, centers, squares)  # 1я лаба
    #
    # scaled_dict = scaleFigures(figarr, squares)  # 2я лаба
    # print("\r80%", end="")
    #
    # swapped_dict = swap_figures(scaled_dict)  # 3я лаба
    # print("\r100%", end="")
    #
    # rotate_left(swapped_dict)  # 3я лаба
    # print("\r110%", end="")
    #
    # rotate_right(swapped_dict)  # 3я лаба
    # print("\r130%", end="")

    ################## 4я лаба

    im = Im.open(im_path_Lab_2)
    imArr = np.asarray(im)
    print("\r140%", end="")

    res = get_rgbsum_image_arr(imArr)
    byteArr, max_bytes_arr = res[0], res[1]
    print("\r150%", end="")

    bin_arr = get_binary_array(byteArr, max_bytes_arr)
    print("\r170%", end="")

    hw = (im.height, im.width)

    reform_to_image(bin_arr, hw)  # 4 Lab
    print("\r190%", end="")

    # Lab_4_impulse.png

    figure_array = getFiguresInImpulseImage(bin_arr)
    #figure_array = getFigures1(bin_arr)
    figure_array = delete_trash(figure_array)

    width_height_arr = get_w_h(figure_array)

    draw_to_image(width_height_arr, imArr)

    print("\r190%", end="")


main()
