from numba import njit, prange
import numpy as np
#from scipy.spatial.qhull import ConvexHull, QhullError
#from dipy.tracking.streamline import set_number_of_points


@njit(fastmath=True)
def dist2(a, b):
    total = np.float32(0)
    for i in range(len(a)):
        d = a[i] - b[i]
        total += d * d
    return total


@njit(fastmath=True)
def clusterize(fibers, threshold):
    N, D = fibers.shape
    threshold *= threshold * (D/3)
    reverser = np.arange(D).reshape((-1, 3))[::-1].flatten()

    # Find a fiber clothest to the center
    center = np.sum(fibers, 0) / N  # mean
    center_rev = center[reverser]
    min_ix = 0
    min_d2 = threshold * 100
    for j in range(N):
        d2 = dist2(center, fibers[j])
        if d2 < min_d2:
            min_ix = j
            min_d2 = d2
        d2 = dist2(center_rev, fibers[j])
        if d2 < min_d2:
            min_ix = j
            min_d2 = d2

    selected_ixs = [min_ix]

    for j in range(N):
        fj = fibers[j]
        fr = fj[reverser]
        # Check the last one first as it's more likely to be a neighbour
        for i in range(len(selected_ixs)-1, -1, -1):
            fi = fibers[selected_ixs[i]]
            if dist2(fj, fi) < threshold:
                break
            if dist2(fr, fi) < threshold:
                break
        else:
            selected_ixs.append(j)

    return selected_ixs


@njit(parallel=True, fastmath=True)
def _classify(fibers, centroids, clabels, threshold, result, is_reversed):
    N, D = fibers.shape
    threshold *= threshold * (D/3)
    reverser = np.arange(D).reshape((-1, 3))[::-1].flatten()

    for i in prange(N):
        f = fibers[i]
        fr = f[reverser]
        min_ix = 0
        min_d2 = threshold
        isr = False

        # Find the closest atlas' fiber
        for j in range(centroids.shape[0]):
            d2 = dist2(f, centroids[j])
            if d2 < min_d2:
                min_ix = j
                min_d2 = d2
            d2 = dist2(fr, centroids[j])
            if d2 < min_d2:
                min_ix = j
                min_d2 = d2
                isr = True

        if min_d2 < threshold:
            result[i] = clabels[min_ix]
        is_reversed[i] = isr


def classify(fibers, centroids, clabels, threshold):
    N, D = fibers.shape
    result = np.zeros(N, dtype=clabels.dtype)
    is_reversed = np.zeros(N, dtype=np.bool8)
    _classify(fibers, centroids, clabels, threshold, result, is_reversed)
    return result, is_reversed
