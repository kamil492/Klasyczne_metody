import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu
import cv2
from skimage import color

## metryki - policzenie poprawnosci dla pojedynczego obrazu
def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union
def iou_score(inputs,target):
    intersection = np.logical_and(inputs, target)
    union = np.logical_or(inputs, target)
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return np.sum(intersection) / np.sum(union)

# wczytanie obrazu
image_org = io.imread("images/val_org.jpg")
image = (color.rgb2gray(io.imread("images/val_org.jpg")))
maska = (color.rgb2gray(io.imread("images/valmask_org.jpg")))
# wywolanie funkcji multiotsu z biblioteki scikit image 
thresholds = threshold_multiotsu(image, classes=2)

#zamiana danych z binarnych na postac cyfrowa
regions = np.digitize(image, bins=thresholds)

# konwersja 64bit na 8 bti z normalizacja etc
img_n = cv2.normalize(src=regions, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

output = img_n.astype(np.uint8)
plt.imshow(output)
plt.imsave("images/walidacyjnyOtsu.jpg", output, cmap='gray')

#stowrzenie subplotu do wyswietlania obrazu oryginalnego, histogramu z naniesionymi liniami i obrazu wynikowego
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 4.5))

#miary poprawnosci
score = dice_metric(regions,maska)
iouscore = iou_score(regions,maska)

# wyswietlenie oryginalu
ax[0].imshow(image_org )
ax[0].set_title('Obraz oryginalny')
ax[0].axis('off')

#wyswietlenie maski
ax[1].imshow(maska, cmap='gray' )
ax[1].set_title('Maska obrazu')
ax[1].axis('off')

# wyswietlenie linii na histogramie
ax[2].hist(image.ravel(), bins=255)
ax[2].set_title('Histogram')
for x in thresholds:
    ax[2].axvline(x, color='r')

# wyswietlenie obrazu wynikowego
ax[3].imshow(output, cmap='gray')
ax[3].set_title('Obraz po progowaniu')
ax[3].axis('off')
plt.subplots_adjust()
plt.show()
