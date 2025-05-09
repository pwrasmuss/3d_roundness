import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy import ndimage


def calculate_euclidian_distance(seg):
    distances = distance_transform_edt(seg)
    return distances


def calculate_maximum_inscribed_spheres(segmentation, label_img,
                                        lower_bound=None):
    if lower_bound is None:
        lower_bound = -1

    labels = np.unique(label_img)
    distance = calculate_euclidian_distance(segmentation)
    max_values = ndimage.labeled_comprehension(distance, label_img,
                                               labels[1:],
                                               np.max, np.float64, None)

    # maximum_inscribed_spheres = []
    maximum_inscribed_spheres = {}
    for idx, label in enumerate(labels[1:]):
        radius = max_values[idx]
        if radius < lower_bound:
            continue
        seg = label_img == label
        centre = np.argwhere(distance*seg == radius)[0]
        maximum_inscribed_spheres[label] = {'radius': radius,
                                            'centre': centre,
                                            'volume': 4/3*np.pi*radius**3
                                            }
        # maximum_inscribed_spheres.append({'label': label,
        #                                   'radius': radius,
        #                                   'centre': centre,
        #                                   'volume': 4/3*np.pi*radius**3
        #                                   })

    return maximum_inscribed_spheres
