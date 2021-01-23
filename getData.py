import numpy as np
import scipy.io as sio
from PIL import Image

load_data = sio.loadmat("old_data.mat")
image = load_data['image'][1,:,:,0]
print(image)
image = Image.fromarray(np.uint8(image*255))

image.save('image.png')