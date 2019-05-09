import os
import matplotlib.pyplot as plt

from skimage import io, exposure
from skimage.feature import hog

# Script params

positive_imgs_dir = "D:/Sunanda/PhD Study/Other Activities/NII_Hackathon/dataset/Positive_mini"
negative_imgs_dir = "D:/Sunanda/PhD Study/Other Activities/NII_Hackathon/dataset/Negative_mini"


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

    if plot is True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('HOG')

    return fd, hog_image


def main():
    print("Loading images")

    positive_imgs_list = read_imgs(positive_imgs_dir)
    negative_imgs_list = read_imgs(negative_imgs_dir)

    # Check
    img = negative_imgs_list[2]
    io.imshow(img)
    # plt.show()

    fd, hog_image = get_hog_img(img, True)

    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
