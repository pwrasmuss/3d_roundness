import numpy as np
from scipy import optimize


def best_sphere(Z_i, r_s, Z_s):
    r_i = r_s**2 - (Z_s - Z_i)**2
    return r_i


def find_best_sphere(centres, label):
    r_s, Z_s = optimize.curve_fit(best_sphere,
                                  xdata=centres[label, 0],
                                  ydata=centres[label, 3],
                                  bounds=[0, np.inf])[0]
    return r_s, Z_s


def separate_concentric_circles(centres, boundary_points, cluster_label=None):
    """Seperates concentric circles into clusts that form independent corners.

    Parameters
    ----------
    centres : array
        Centres of all corners found.
    cluster_label : array
        Labels from the DBSCAN clustering.
    boundary_points : array
        Boundary points associated with the corners.

    Returns
    -------
    np.array([Z_s, x_mean, y_mean, r_s, slice_label, fat_label])
    """
    spheres = []
    if cluster_label is None:
        centre_indices = np.arange(centres.shape[0])
    else:
        centre_indices = np.argwhere(cluster_label == True)[:, 0]

    count = 0
    start = 0
    end = centre_indices.size

    num_centres_used = centre_indices[start:end].size

    while True:
        if start == end:
            return spheres
        # Calculates the best fitting sphere in corner based on all points in a group.
        r_s, Z_s = find_best_sphere(centres, centre_indices[start:end])
        # print(r_s, Z_s)
        # Calculates the centre of the circle sphere
        x_mean = np.mean(centres[centre_indices[start:end], 2])
        y_mean = np.mean(centres[centre_indices[start:end], 1])

        # Selects boundary points
        selected_boundary_points = boundary_points[centre_indices[start:end]]
        sphere_centre = np.asarray([Z_s, x_mean, y_mean])

        minimum_distance = 1e9
        for idx, points in enumerate(selected_boundary_points):
            # Calculate distance from best fitting sphere to boundary points
            dist = np.sqrt(np.sum((points - sphere_centre)**2, axis=1))
            # Save the minimum value
            minimum_distance = min(minimum_distance, dist.min())

        # If the sphere is within the boundary we're good
        if r_s < minimum_distance:
            # print('\tGood')
            # Final entry is label
            slice_label = centres[start, -2]
            fat_label = centres[start, -1]
            spheres.append(np.array([Z_s, x_mean, y_mean, r_s, slice_label, fat_label]))

            # If count == 0 all centres are used and we return
            if count == 0:
                return spheres
            # If not we exclude previously used centres and continue
            end = start
            start = 0
            num_centres_used = centre_indices[start:end].size
            count = 0

        elif r_s > minimum_distance:
            # If the condition is not met and we have two centres in total return
            if centre_indices.size == 2:
                print('Points could not solve corner eq.')
                return spheres

            start += 1
            num_centres_used = centre_indices[start:end].size
            if num_centres_used < 2:
                start = 0
        count += 1
        # If we have excluded everything but the final two centres
        if count == centre_indices.size - 2:
            start = 0
            end -= 1
            count = 0
