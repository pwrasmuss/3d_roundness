# Credits: Deepwings project
# Source: https://github.com/machine-shop/deepwings/blob/master/deepwings/method_features_extraction/image_processing.py#L156-L245
# No license specified, MIT license assumed

import numpy as np


def moore_neighborhood(current, backtrack):  # y, x
    """Returns clockwise list of pixels from the moore neighborhood of current
    pixel:
    The first element is the coordinates of the backtrack pixel.
    The following elements are the coordinates of the neighboring pixels in
    clockwise order.

    Parameters
    ----------
    current ([y, x]): Coordinates of the current pixel
    backtrack ([y, x]): Coordinates of the backtrack pixel

    Returns
    -------
    List of coordinates of the moore neighborood pixels, or 0 if the backtrack
    pixel is not a current pixel neighbor
    """

    operations = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    )
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def boundary_tracing(region, im_shape, warnings=False):
    """Coordinates of the region's boundary. The region must not have isolated
    points.

    Parameters
    ----------
    region : obj
        Obtained with skimage.measure.regionprops()

    Returns
    -------
    boundary : 2D array
        List of coordinates of pixels in the boundary
        The first element is the most upper left pixel of the region.
        The following coordinates are in clockwise order.
    """

    # creating the binary image
    coords = region.coords
    row_limit = ((coords[:, 0] == 0).any() or (coords[:, 0] == (im_shape[0] - 1)).any())
    col_limit = ((coords[:, 1] == 0).any() or (coords[:, 1] == (im_shape[1] - 1)).any())
    if row_limit or col_limit:
        if warnings:
            print('Region is on the border.')
        return None
    # print('coords', coords)
    maxs = np.max(coords, axis=0)
    # print('maxs', maxs)
    # Padding to avoid error when calculating focus_start
    binary = np.zeros((maxs[0] + 3, maxs[1] + 3))
    # print('binary.shape', binary.shape)
    x = coords[:, 1] + 1
    # print('x', x)
    y = coords[:, 0] + 1
    # print('y', y)
    binary[tuple([y, x])] = 1
    # print('binary', binary)

    # initilization
    # starting point is the most upper left point
    idx_start = 0
    while True:  # asserting that the starting point is not isolated
        start = [y[idx_start], x[idx_start]]
        # print('start', start)
        focus_start = binary[start[0]-1: start[0]+2,
                             start[1]-1: start[1]+2]
        # print('focus_start', focus_start)
        # print(start[0]-1, start[0]+2)
        # print(start[1]-1, start[1]+2)

        if np.sum(focus_start) > 1:
            break
        idx_start += 1

        if idx_start == len(x):
            if warnings:
                print('Starting point might be isolated.')
            return None

    # Determining backtrack pixel for the first element
    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0

    # print('start/backtrack_start:', start, backtrack_start)
    while True:
        # print(current, backtrack)
        neighbors_current = moore_neighborhood(current, backtrack)
        # print('neighbors_current', neighbors_current)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        # print('argmax = idx: ', binary[tuple([y, x])])
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx - 1]
        current = neighbors_current[idx]
        counter += 1

        # print('current/start:', current, start, '\t',
        #       'backtrack/backtrack_start:', backtrack, backtrack_start)

        if counter > len(coords):
            if warnings:
                print("Boundary couldn't be ordered in a reasonable number of steps.")
            return None

        if np.all(current == start) and np.all(backtrack == backtrack_start):
            # print(current, backtrack)
            # print(f'break {counter}\n')
            break

        # print()

    return np.array(boundary) - 1
