import os
import matplotlib.pyplot as plt

from skimage import io, exposure
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray

from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split


# Script params

positive_imgs_dir = "D:/Sunanda/PhD Study/Other Activities/NII_Hackathon/dataset/Positive_mini"
negative_imgs_dir = "D:/Sunanda/PhD Study/Other Activities/NII_Hackathon/dataset/Negative_mini"
pca_components = 5


def prepare_datasets(positives, negatives):
    y_pos = np.ones(len(positives))
    y_neg = np.zeros(len(negatives))
    y = np.concatenate((y_pos, y_neg))

    X = positives + negatives

    X_train, X_remain, y_train, y_remain = train_test_split(X, y, stratify=y, test_size=0.40)

    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, stratify=y_remain, test_size=0.50)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def read_imgs(dir):
    imgs_list = []

    directory = os.fsencode(dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(dir, filename)
            img = io.imread(path)
            imgs_list.append(img)

    return imgs_list


def get_hog_img(img, plot=False):
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    if plot is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('HOG')

        hog_image_rescaled = rgb2gray(hog_image_rescaled)

    return fd, hog_image_rescaled


def down_sample(img, factor):
    x = img.shape[0] // factor
    y = img.shape[1] // factor

    img_resized = resize(img, (x, y), anti_aliasing=False)

    return img_resized


def do_pca(X, n_components=None, pca_model=None):
    print('Doing PCA. n_components={}'.format(n_components))

    if pca_model is None:
        pca_model = PCA(n_components=n_components).fit(X)
        var_ratio = pca_model.explained_variance_ratio_
        cum_var_ratio = np.cumsum(var_ratio)
        print('Did PCA. explained_variance_ratio and cumulative sum is given below\n{}\n{}'.format(var_ratio, cum_var_ratio))
        print('PCA. reduced_num_of_dimensions={}, variance_ratio accounted for={}'.format(len(cum_var_ratio), cum_var_ratio[-1]))

    X_reduced = pca_model.transform(X)

    return pca_model, X_reduced


def get_hog_imgs_list(imgs_list):
    hog_imgs_list = [get_hog_img(img)[1]for img in imgs_list]

    for img in imgs_list:
        fd, hog_img = get_hog_img(img)
        hog_imgs_list.append()

    return hog_imgs_list



def main():
    print("Loading images")

    positive_imgs_list = read_imgs(positive_imgs_dir)
    negative_imgs_list = read_imgs(negative_imgs_dir)


    # Check
    img = positive_imgs_list[2]
    # io.imshow(img)
    fd, hog_image = get_hog_img(img, True)
    # plt.show()

    # Downsample hog_image (reduces sharpness)
    # ds_hog_img = down_sample(hog_image, 2)
    # plt.figure()
    # io.imshow(ds_hog_img, cmap=plt.cm.gray)

    hog_positives = [get_hog_img(img)[1] for img in positive_imgs_list]
    hog_negatives = [get_hog_img(img)[1] for img in negative_imgs_list]

    hog_positives = [hog_img.flatten() for hog_img in hog_positives]
    hog_negatives = [hog_img.flatten() for hog_img in hog_negatives]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets(hog_positives, hog_negatives)

    # Do PCA
    pca_model, X_train = do_pca(X_train, n_components=pca_components)
    unused_var, X_val = do_pca(X_val, pca_model=pca_model)
    unused_var, X_test = do_pca(X_test, pca_model=pca_model)

    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
