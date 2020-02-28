from colour import Color
import random


def get_colors(n):
    """

    :param n: the number of color to return
    :return: a list of colors with a length of n
    """
    colors = [Color('#' + ''.join([random.choice('0123456789abcdef') for j in range(6)])) for i in
              range(n)]
    colors_rgb = list(map(lambda color: [int(255 * c) for c in list(color.get_rgb())], colors))
    for i in range(len(colors_rgb)):  # Make a light color
        m = min(colors_rgb[i])
        M = max(colors_rgb[i])
        if M <= 120:  # If the color is too dark
            for j in range(3):
                if colors_rgb[i][j] == M:
                    colors_rgb[i][j] = min(50 + 3 * colors_rgb[i][j], 255)
                elif colors_rgb[i][j] == m:
                    colors_rgb[i][j] = min(10 + int(1.5 * colors_rgb[i][j]), 255)
                else:
                    colors_rgb[i][j] = min(25 + 2 * colors_rgb[i][j], 255)
    return colors_rgb


