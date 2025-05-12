import edt
from wadell_rs_local import common
from wadell_rs_local import roundness
import concentric_circles
import numpy as np
from skimage import measure
from sklearn.cluster import DBSCAN


def prepare_data(vol_ZXY):
    """
    Creates the volumes needed to calculate the 3D roundness.

    Arguments:
    vol_ZXY ([z, x, y]): np.array
        A binary segmentation of a volume where the background
        is labelled 0 and the objects of interest are labelled 1.

    Returns:
    volumes: list
        A list that contains the input volume and two rotated volumes
        Z -> X and Y -> Z.
    volume_labels: list
        A list that contains the labelled version of the volume list. Each
        object volume has a unique label.
    slice_labels: list
        A list that contains the labelled version of the volume list. Each
        object in each slice of the volumes has a unique label.
    """
    vol_XZY = np.moveaxis(np.copy(vol_ZXY), 1, 0)  # XZY
    vol_YXZ = np.moveaxis(np.copy(vol_ZXY), 2, 0)  # YXZ
    volume_label_ZXY = measure.label(vol_ZXY)
    volume_label_XZY = np.moveaxis(np.copy(volume_label_ZXY), 1, 0)  # XZY
    volume_label_YXZ = np.moveaxis(np.copy(volume_label_ZXY), 2, 0)  # YXZ

    volumes = [vol_ZXY, vol_XZY, vol_YXZ]
    volume_labels = [volume_label_ZXY, volume_label_XZY, volume_label_YXZ]

    slice_label_ZXY = np.empty_like(volume_label_ZXY)
    slice_label_XZY = np.empty_like(volume_label_XZY)
    slice_label_YXZ = np.empty_like(volume_label_YXZ)

    for iSlice in range(vol_ZXY.shape[0]):
        slice_label_ZXY[iSlice] = measure.label(vol_ZXY[iSlice])
        slice_label_XZY[iSlice] = measure.label(vol_XZY[iSlice])
        slice_label_YXZ[iSlice] = measure.label(vol_YXZ[iSlice])
    slice_labels = [slice_label_ZXY, slice_label_XZY, slice_label_YXZ]

    return volumes, volume_labels, slice_labels


def create_label_map(volume_labels, slice_labels):
    """
    Creates a map between volume_labels and slice_labels.

    Arguments:
    volume_labels: list
        A list that contains the labelled version of the volume list. Each
        object volume has a unique label.
    slice_labels: list
        A list that contains the labelled version of the volume list. Each
        object in each slice of the volumes has a unique label.

    Returns:
    label_map: list
        A list that creates a correspondence between volume_labels and
        slice_labels. The list contains three (number of spatial directions)
        dictionaries which contain N dictionaries (number of slices in vol)
        that contain a dictionary for each label in a given slice.
    """
    label_map = []
    for i in range(len(slice_labels)):
        label_map.append({})
        for z in range(volume_labels[0].shape[0]):
            unique_2d_labels = np.unique(slice_labels[i][z, :, :])
            label_map[i][z] = {}
            for lbl in unique_2d_labels:
                if lbl == 0:  # Ignore background
                    continue
                # Find corresponding 3D labels
                mask_2d = slice_labels[i][z, :, :] == lbl
                unique_3d_labels = np.unique(volume_labels[i][z, :, :][mask_2d])
                # Ignore background
                unique_3d_labels = unique_3d_labels[unique_3d_labels > 0]

                if len(unique_3d_labels) == 1:
                    label_map[i][z][lbl] = unique_3d_labels[0]
                else:
                    print('This never happens')
    return label_map


def calc_2d_roundness(vol_numpy, label_img, start=0, end=None, step=1,
                      return_obj=False, warnings=False):
    """
    Calculates the 2D roundness of all objects in all slices in a volume.

    Arguments:
    vol_numpy (float np.array [z, x, y]):
        A binary segmentation of a volume where the background
        is labelled 0 and the objects of interest are labelled 1.
    label_img (float np.array [z, x, y]):
        A labelled version of vol_numpy where each object in each slice of the
        volume is labelled with a unique label.
    start=0 (int):
        The initial slice that is analysed.
    end=None (int):
        The final slice that is analysed. The default value is the last slice
        of the volume.
    step=1 (int):
        steps between slices.

    return_obj=False (bool):
        returns roundness_values and obj_dict_lists if set to True

    Returns:
    roundness_values (list of dicts):
        Has a length range(start, end, step) and contains dictionaries with
        the objects slice_label and its roundness_value.
    """
    max_dev_thresh = 0.5  # Maximum deviation from a straight line for discretization
    circle_fit_thresh = 0.98  # Defines how close the corner points have to be to the fitted circle outline
    smoothing_method = "energy"  # Method for smoothing the boundary, 'energy' or 'loess'
    alpha_ratio = 0.001  # Ratio of the energy term for the boundary length
    beta_ratio = 0.001  # Ratio of the energy term for the boundary curvature

    if end is None:
        end = vol_numpy.shape[0]

    obj_dict_lists = []
    roundness_values = []
    for iSlice in range(start, end, step):
        if warnings:
            print(f'Slice number {iSlice}')
        roundness_values.append([])
        edt_img = edt.edt(vol_numpy[iSlice])  # Calculate distance transform

        obj_dict_list = common.characterize_objects(label_img[iSlice], edt_img,
                                                    warnings=warnings)  # Collect characteristics about binary objects
        obj_dict_lists.append(obj_dict_list)
        for obj_dict in obj_dict_list:
            roundness_value = roundness.calculate_roundness(
                    obj_dict,
                    max_dev_thresh,
                    circle_fit_thresh,
                    smoothing_method=smoothing_method,
                    alpha_ratio=alpha_ratio,
                    beta_ratio=beta_ratio,
                    verbose=True,
                )
            roundness_values[iSlice].append({'roundness_value': roundness_value,
                                             'slice_label': obj_dict['slice_label']})

    if return_obj:
        return roundness_values, obj_dict_lists
    return roundness_values


def get_centres_and_boundaries(roundness_values, label_map, start=0, end=None, step=1):
    """
    Gets the centres and the boundaries of the corner circles for a list of
    roundness values.

    Args:
        roundness_values (list): _description_
        label_map (list): label map between slice labels and global labels.
        start (int, optional): The initial slice. Defaults to 0.
        end (int, optional): The final slice. Defaults to None. If None end
        equals len(roundness_values).
        step (int, optional): steps between slices. Defaults to 1.

    Returns:
        centres (list of tuples): List of centres of corner circles in the
        slices in the volumes.
        boundary_points (list of np.array): List of centres of corner circles
        in the slices in the volumes.
    """

    if end is None:
        end = len(roundness_values)
    centres = []
    boundary_points = []
    for iSlice, roundness_value in zip(range(start, end, step), roundness_values):
        for _roundness_value in roundness_value:
            for centre, radius in zip(_roundness_value['roundness_value'][1]['corner_circle_centers'], _roundness_value['roundness_value'][1]['corner_circle_radii']):
                # print(iSlice, _roundness_value['slice_label'])
                volume_label = label_map[iSlice][_roundness_value['slice_label']]
                # volume_label = -99
                centres.append(np.asarray([iSlice, centre[0], centre[1], radius, _roundness_value['slice_label'], volume_label]))

                points_3D = np.hstack((np.full((_roundness_value['roundness_value'][1]['boundary_points'].shape[0], 1), iSlice),
                                      _roundness_value['roundness_value'][1]['boundary_points']))
                boundary_points.append(points_3D)
    centres = np.asarray(centres)
    boundary_points = np.asarray(boundary_points, dtype=object)

    return centres, boundary_points


def calculate_all_spheres(centres, boundary_points, eps=0.9):
    """
    Clusters concentric circles and fits spheres to them.
    Args:
        centres (list of np.array): List of centres of corner circles in the
        slices in the volumes.
        boundary_points (list of np.array): List of centres of corner circles
        in the slices in the volumes.
        eps (float, optional): Cluster limit for DBSCAN. Defaults to 0.9.

    Returns:
        spheres (_type_): _description_
    """
    spheres = []
    unique_vol_labels = np.unique(centres[:, -1])
    for unique_label in unique_vol_labels:
        index = centres[:, -1] == unique_label
        tmp = centres[index]
        db = DBSCAN(eps=eps, min_samples=2).fit(tmp[:, 1:-3])
        cluster_labels = db.labels_

        if cluster_labels.max() == -1:
            continue
        for cluster_label in range(cluster_labels.max()+1):
            sphere = concentric_circles.separate_concentric_circles(tmp[cluster_labels == cluster_label],
                                                                    boundary_points)
            spheres.append(sphere)
    return spheres


def reorder_coordinates(arr, i):
    """
    Reorders the centre coordinates of a sphere such that it matches the
    coordinate order ZXY.

    Args:
        arr (float np.array): Numpy array that contains information about a
        sphere. The first three indices denote the centre of that sphere.
        i (int): An index between 0 and 2 that controls how coordinates are
        moved. np.array([Z_s, x_mean, y_mean, r_s, slice_label, fat_label])

    Raises:
        ValueError: If i is anything else than 0, 1 or 2.

    Returns:
        arr (float np.array): returns arr with reordered centre cordinates.
        np.array([Z_s, x_mean, y_mean, r_s, slice_label, fat_label])
    """
    if i == 0:
        return arr
    elif i == 1:
        return np.array([arr[1], arr[0], arr[2], arr[3], arr[4], arr[5]])
    elif i == 2:
        return np.array([arr[2], arr[1], arr[0], arr[3], arr[4], arr[5]])
    else:
        raise ValueError()


def flatten_spheres(spheres_in_vol):
    """
    Flattens the input list into a single array.

    Args:
        spheres_in_vol (list of float np.array): list of spheres

    Returns:
        flat_spheres (float np.array): np.array of spheres
        np.array([Z_s, x_mean, y_mean, r_s, slice_label, fat_label])
    """
    flat_spheres = []
    for i, sphere_in_vol in enumerate(spheres_in_vol):
        for sphere in sphere_in_vol:
            if len(sphere) == 0:
                continue
            _sphere = reorder_coordinates(sphere[0], i)
            flat_spheres.append(_sphere)
    flat_spheres = np.array(flat_spheres)

    return flat_spheres


def merge_duplicate_spheres(spheres, centre_tolerance=0.1, radius_tolerance=0.05):
    """
    Merges duplicate spheres by averaging their properties.

    :param spheres: (N, 6) array of spheres
    :param centre_tolerance: Distance threshold for considering centres as duplicates
    :param radius_tolerance: Radius difference threshold for duplicates
    :return: spheres after merging duplicates
    """

    # Use DBSCAN to cluster similar spheres
    clustering = DBSCAN(eps=centre_tolerance, min_samples=2).fit(spheres[:, :3])

    labels = clustering.labels_
    unique_labels = set(labels)

    merged_spheres = []

    processed = set()

    for label in unique_labels:
        if label == -1:
            continue  # Noise (not part of any cluster)

        # Get indices of all spheres in this cluster
        indices = np.where(labels == label)[0]

        # Further filter by radius similarity
        mean_radius = np.mean(spheres[:, 3][indices])
        close_radii = indices[np.abs(spheres[:, 3][indices] - mean_radius) < radius_tolerance]

        if len(close_radii) > 1:
            # Average the positions and radii
            avg_sphere = np.mean(spheres[close_radii], axis=0)

            merged_spheres.append(avg_sphere)
            processed.update(close_radii)

    # Keep unprocessed spheres
    unprocessed = [i for i in range(spheres.shape[0]) if i not in processed]

    merged_spheres.extend(spheres[unprocessed])

    return np.array(merged_spheres)
