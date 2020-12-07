import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)


# resize TRAIN images
input_folder = "./input/siim-isic-melanoma-classification/jpeg/train/"
output_folder = "./input/kaggle/working/train224"
images = glob.glob(os.path.join(input_folder, "*.jpg"))

Parallel(n_jobs=12)(
    delayed(resize_image)
        (i, output_folder, (224, 224)) for i in tqdm(images)
)

# resize TEST images
input_folder = "./input/siim-isic-melanoma-classification/jpeg/test/"
output_folder = "./input/kaggle/working/test224"
images = glob.glob(os.path.join(input_folder, "*.jpg"))

Parallel(n_jobs=12)(
    delayed(resize_image)
        (i, output_folder, (224, 224)) for i in tqdm(images)
)
