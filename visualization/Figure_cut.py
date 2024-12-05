from PIL import Image
import numpy as np


def crop_image(image_path, output_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    img_array = np.array(img)

    r, g, b, alpha = img_array[..., 0], img_array[..., 1], img_array[..., 2], img_array[..., 3]

    mask = (alpha > 0) & ~((r == 255) & (g == 255) & (b == 255))

    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped_img.save(output_path)

image_name = "Vor3"
image_path = image_name + ".png"
output_path = image_name + "_cut.png"
crop_image(image_path, output_path)
