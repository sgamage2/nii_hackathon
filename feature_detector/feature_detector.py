import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

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
            img = Image.open(path)
            imgs_list.append(img)

    return imgs_list


def main():
    print("Loading images")

    positive_imgs_list = read_imgs(positive_imgs_dir)
    negative_imgs_list = read_imgs(negative_imgs_dir)

    # negative_imgs_list[0].show()

    print("Done")


if __name__ == "__main__":
    main()
