import sys
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

plt.rcParams.update({'axes.labelsize':10, 
                     'xtick.labelsize':6, 
                     'ytick.labelsize':6, 
                    })

# Open image file
img1_dir = sys.argv[1]
img2_dir = sys.argv[2]

img1_pil = pilimg.open(img1_dir)
img2_pil = pilimg.open(img2_dir)

img1 = np.array(img1_pil)
img2 = np.array(img2_pil)

# Get input for rotate and scale
print('Enter list of rotate (counter-clockwise) degree in integer number (default: [0]): ', end='')
rotate = list(map(int, input().split()))
print('Enter list of scale factor in positive float (default: [1.0]): ', end='')
scale = list(map(float, input().split()))

if len(rotate)==0:
    rotate = [0]
if len(scale)==0:
    scale=[1.0]

# Check image shape mismatch
if img1.shape[2]!=img2.shape[2]:
    print("Image channel mismatch!")
    
if (img1.shape[0]-img2.shape[0])*(img1.shape[1]-img2.shape[1])<0:
    print("Image shape mismatch!")

# Determine image size
img_small = None
img_large = None
img_small_pil = None
if img1.shape[0]<img2.shape[0]:
    img_small = img1
    img_large = img2
    img_small_pil = img1_pil
else:
    img_small = img2
    img_large = img1
    img_small_pil = img2_pil
img_reference_original = img_large.copy()

## Preprocessing

# Convert image data type
img_small = img_small.astype(np.int32)
img_large = img_large.astype(np.int32)

# Define image width and height
height_small = img_small.shape[0]
width_small = img_small.shape[1]
height_large = img_large.shape[0]
width_large = img_large.shape[1]
depth = img_small.shape[2]

# Define variables
scaled_size_list = []
for s in scale:
    scaled_size_list.append((int(height_small*s), int(width_small*s)))
scale_len = len(scale)
rotate_len = len(rotate)

# Apply scaling and rotation to template image and show transformed images
fig, axes = plt.subplots(scale_len, rotate_len, figsize=(2*rotate_len,2*scale_len), squeeze=False, gridspec_kw={'height_ratios':[h for (h,w) in scaled_size_list]})
img_small_trans = []
for si, s in enumerate(scale):
    temp_trans = np.zeros((rotate_len, scaled_size_list[si][0], scaled_size_list[si][1], depth), dtype=np.int32)
    for ri, r in enumerate(rotate):
        temp_trans[ri] = np.array(img_small_pil.resize(scaled_size_list[si]).rotate(r, fillcolor=(img_small[0,0,0], img_small[0,0,1], img_small[0,0,2])))
        axes[si,ri].imshow(temp_trans[ri])
        axes[si,ri].set_title(f'scale:{s} rotate:{r}', fontsize=8)
    img_small_trans.append(temp_trans)
fig.suptitle('transformed template image', fontsize=8)
plt.show(block=False)

img_large = img_large - np.mean(img_large, dtype=np.float64)
for si in range(scale_len):
    for ri in range(rotate_len):
        img_small_trans[si][ri] = img_small_trans[si][ri] - np.mean(img_small_trans[si][ri], dtype=np.float64)

        
# Compute Normalized Cross-Correlation (NCC)
corr_result = []
for si in range(scale_len):
    corr_result.append(np.zeros((rotate_len, height_large-scaled_size_list[si][0]+1, width_large-scaled_size_list[si][1]+1), dtype=np.float64))
img_small_norm_list = np.zeros((scale_len, rotate_len), dtype=np.float64)
for si in range(scale_len):
    for ri in range(rotate_len):
        img_small_norm_list[si,ri] = np.linalg.norm(img_small_trans[si][ri].reshape((-1)))
for row in tqdm(range(height_large)):
    for col in range(width_large):
        for si in range(scale_len):
            if row+scaled_size_list[si][0] > height_large or col+scaled_size_list[si][1] > width_large:
                continue
            img_large_crop = img_large[row:row+scaled_size_list[si][0], col:col+scaled_size_list[si][1], :]
            img_large_crop_norm = np.linalg.norm(img_large_crop.reshape((-1)))
            for ri in range(rotate_len):
                corr_result[si][ri, row, col] = np.sum(img_large_crop*img_small_trans[si][ri], dtype=np.float64) / (img_large_crop_norm * img_small_norm_list[si, ri])

# Plot result array as image
fig, axes = plt.subplots(scale_len, rotate_len, figsize=(4*rotate_len,4*scale_len), squeeze=False)
for si, s in enumerate(scale):
    for ri, r in enumerate(rotate):
        tax = axes[si,ri].imshow(corr_result[si][ri], cmap=cm.rainbow, vmin=-1., vmax=1.)
        plt.colorbar(tax, ax=axes[si,ri])
        axes[si,ri].set_title(f'scale:{s} rotate:{r}', fontsize=8)
fig.suptitle('NCC result', fontsize=16)
plt.show(block=False)


# Plot red box where ncc result values are larger than threshold
threshold = 0.8
img_reference_boxed = np.zeros((height_large, width_large), dtype=np.bool_)
fig, axes = plt.subplots(1, 1, squeeze=False)
axes[0,0].imshow(img_reference_original)
for si in range(scale_len):
    height_small_curr, width_small_curr = scaled_size_list[si]
    for ri in range(rotate_len):
        y_coord, x_coord = np.unravel_index(np.argsort(corr_result[si][ri].reshape((-1)))[::-1], corr_result[si][ri].shape)
        y_coord = y_coord[:len(np.where(corr_result[si][ri]>threshold)[0])]
        x_coord = x_coord[:len(np.where(corr_result[si][ri]>threshold)[0])]
        for x, y in zip(x_coord, y_coord):
            if img_reference_boxed[y,x] \
                or img_reference_boxed[y,x+width_small_curr] \
                or img_reference_boxed[y+height_small_curr,x] \
                or img_reference_boxed[y+height_small_curr,x+width_small_curr]:
                continue
            axes[0,0].plot((x, x), (y, y+height_small_curr), '-', color='red')
            axes[0,0].plot((x+width_small_curr, x+width_small_curr), (y, y+height_small_curr), '-', color='red')
            axes[0,0].plot((x, x+width_small_curr), (y, y), '-', color='red')
            axes[0,0].plot((x, x+width_small_curr), (y+height_small_curr, y+height_small_curr), '-', color='red')
            img_reference_boxed[y:y+height_small_curr, x:x+width_small_curr] = True
axes[0,0].set_title(f'Template Matching Result (threshold={threshold})')
plt.savefig(f'./result/ncc_result_{"".join(img1_dir.split("/")[-1].split(".")[:-1])}_{"".join(img2_dir.split("/")[-1].split(".")[:-1])}.png', dpi=400)
plt.show()
