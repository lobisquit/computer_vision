import warnings
from argparse import ArgumentParser

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import ConvexHull
from skimage.filters import gaussian, laplace
from skimage.io import imread, imsave, imshow, show
from skimage.morphology import closing, dilation, opening
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)

# add parser for script working modes
parser = ArgumentParser()
parser.add_argument("--mode", dest="mode",
                    choices=['show_all', 'show_errors', 'save'],
                    default='show_all',
                    help="Either show or save images")
args = parser.parse_args()

# ground truth
true_Ks = [0, 1, 3, 3,
           2, 2, 2, 3,
           4, 2, 4, 3,
           4, 3, 2, 2,
           3, 3]

# paint each cluster of a different color
cluster_colors = {
    0: np.array([216, 39, 39]) / 255,
    1: np.array([216, 201, 39]) / 255,
    2: np.array([115, 216, 39]) / 255,
    3: np.array([39, 216, 168]) / 255,
    4: np.array([39, 130, 216]) / 255,
    5: np.array([71, 39, 216]) / 255,
    6: np.array([216, 39, 139]) / 255
}

def read(image_id, kind):
    '''
    Read image corresponding to image_id, either of kind "color" or "depth"
    '''
    image = imread('dataset/{}{}.png'.format(kind, image_id))

    return image / np.max(image)

# read image with no people, as environment reference
none_depth = read(0, 'depth')

correct = 0
for image_id in range(1, 18):
    print('Doing {}/{}'.format(image_id, len(true_Ks) - 1), end='\r')

    color_image = read(image_id, 'color')
    depth_image = read(image_id, 'depth')

    # remove environment
    depth_image = none_depth - depth_image

    # rescale image pixels in [0, 1]
    minimum = np.min(depth_image)
    maximum = np.max(depth_image)
    depth_image = (depth_image - minimum) / (maximum - minimum)

    # remove border noise: detect high derivative points of different
    # sign and close to each other with second order derivative
    first_derivative = laplace(depth_image)
    second_derivative = laplace(first_derivative)

    # threshold both first and second order derivatives, creating a mask
    first_mask = np.asarray(np.abs(second_derivative) > 0.75)
    second_mask = np.asarray(np.abs(first_derivative) > 0.25)
    mask = np.asarray(np.logical_and(first_mask, second_mask), dtype=np.float)

    # consider mask points neighbours, to remove noise completely
    noisy_borders = dilation(mask) == 1
    depth_image[noisy_borders] = np.mean(depth_image)

    # create (row, col, color) data
    rows, cols = np.mgrid[0:depth_image.shape[0],
                          0:depth_image.shape[1]]
    rows = rows.ravel(order='F')
    cols = cols.ravel(order='F')
    values = depth_image.ravel(order='F')
    points = np.array(list(zip(rows, cols, values)))

    # filter out background points, to speed up clustering
    points = points[points[:, 2] > 0.6]

    # save clustering and score for various number of clusters
    # in order to assess the best one
    Ks = list(range(1, 6))
    scores = []
    clusters_candidates = []

    # test various number of clusters
    for K in Ks:
        cl = KMeans(n_clusters=K, n_jobs=-2)
        clusters = cl.fit_predict(points)

        clusters_candidates.append(clusters)

        # use mean square distance from its centroid for each point
        # in the dataset, penalizing fragmented clusterings
        score = np.log(cl.inertia_) + K * 0.4
        scores.append(score)

    # find clustering with best score
    best_index = np.argmin(scores)

    best_K = Ks[best_index]
    best_clustering = clusters_candidates[best_index]
    points[:, 2] = best_clustering

    # mark clusters in original (color) image
    for x, y, cluster in points:
        col = int(x)
        row = int(y)
        cluster_id = int(cluster)

        color_image[col, row] = 0.5 * color_image[col, row] + \
                                0.5 * cluster_colors[cluster_id]

    # report errors
    error = best_K != true_Ks[image_id]
    if error:
        print('In {}, found {} people, {} expected'.format(image_id,
                                                           best_K,
                                                           true_Ks[image_id]))

        # for n, score in zip(range(1, 5), scores):
            # print('n={} => score={}'.format(n, score))
    else:
        correct += 1

    if args.mode == 'show_all':
        imshow(color_image)

    elif args.mode == 'show_errors' and error:
        imshow(color_image)

    elif args.mode == 'save':
        imsave('results/detection-{}.png'.format(image_id), color_image)

    show()

print('{}/{} are correct'.format(correct, len(true_Ks) - 1))
