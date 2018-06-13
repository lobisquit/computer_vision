from argparse import ArgumentParser

import numpy as np

import cv2 as cv

# add parser for script working modes
parser = ArgumentParser()
parser.add_argument("--mode", dest="mode",
                    choices=['show_all', 'show_errors', 'save'],
                    default='show_all',
                    help="Either show or save images")
parser.add_argument("--eta", dest="eta",
                    type=float,
                    default=0.42,
                    help="Regularization parameter")
args = parser.parse_args()

# ground truth
true_Ks = [0, 1, 3, 3,
           2, 2, 2, 3,
           4, 2, 4, 3,
           4, 3, 2, 2,
           3, 3]

# paint each cluster of a different color (B, G, R) format
cluster_colors = {
    0: np.array([39, 39, 216]),
    1: np.array([39, 201, 216]),
    2: np.array([39, 216, 115]),
    3: np.array([168, 216, 39]),
    4: np.array([216, 130, 39]),
    5: np.array([216, 39, 71]),
    6: np.array([139, 39, 216])
}

def read(image_id, kind):
    '''
    Read image corresponding to image_id, either of kind "color" or "depth"
    '''
    path = 'dataset/{}{}.png'.format(kind, image_id)

    if kind == 'depth':
        # convert image to float automatically
        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        image = np.float32(image)
        image /= np.max(image)
    elif kind == 'color':
        # convert image to float automatically
        image = cv.imread(path)

    return image

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
    first_derivative = cv.Laplacian(depth_image, cv.CV_32F, 3)
    second_derivative = cv.Laplacian(first_derivative, cv.CV_32F, 3)

    # threshold both first and second order derivatives, creating a mask
    first_mask = np.asarray(np.abs(second_derivative) > 0.75)
    second_mask = np.asarray(np.abs(first_derivative) > 0.25)
    mask = np.asarray(np.logical_and(first_mask, second_mask), dtype=np.float32)

    # consider mask points neighbours, to remove noise completely
    noisy_borders = cv.dilate(mask, np.ones((3, 3)), iterations=1)
    depth_image[noisy_borders == 1.] = np.mean(depth_image)

    # create (row, col, color) data
    rows, cols = np.mgrid[0:depth_image.shape[0],
                          0:depth_image.shape[1]]
    rows = rows.ravel(order='F')
    cols = cols.ravel(order='F')
    values = depth_image.ravel(order='F')
    points = np.array(list(zip(rows, cols, values)))

    # filter out background points, to speed up clustering
    points = points[points[:, 2] > 0.6]
    points = np.float32(points)

    # save clustering and score for various number of clusters
    # in order to assess the best one
    Ks = list(range(1, 6))
    scores = []
    clusters_candidates = []

    # set Kmeans parameters once for all
    accuracy = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
    max_iters = 10
    convergence_type = 1.0

    Kmeans_params = {
        'criteria': (accuracy, max_iters, convergence_type),
        'attempts': 10,
        'flags': cv.KMEANS_PP_CENTERS,
        'bestLabels': None
    }

    # test various number of clusters
    for K in Ks:
        inertia, labels, centers = cv.kmeans(points, K, **Kmeans_params)

        clusters_candidates.append(labels)

        # use mean square distance from its centroid for each point
        # in the dataset, penalizing fragmented clusterings
        score = np.log(inertia) + K * args.eta
        scores.append(score)

    # find clustering with best score
    best_index = np.argmin(scores)

    best_K = Ks[best_index]
    best_clustering = clusters_candidates[best_index]
    points[:, 2] = best_clustering.reshape( (-1,) )

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
    else:
        correct += 1

    if args.mode == 'show_all' or (args.mode == 'show_errors' and error):
        window_name = 'image {}'.format(image_id)
        cv.imshow(window_name, color_image)

        cv.waitKey(0)
        cv.destroyWindow(window_name)

    elif args.mode == 'save':
        cv.imwrite('results/detection-{}.png'.format(image_id), color_image)

print('{}/{} are correct'.format(correct, len(true_Ks) - 1))
