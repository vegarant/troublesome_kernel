import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat

idx = 968;

fname = f'sample_{idx}.pt'

data = torch.load(fname);

print(data.dtype)

data = data.numpy()
print('np.amax(data): ', np.amax(data))

im = Image.fromarray(np.uint8(np.round(255*data)));

fsize = 13;

draw = ImageDraw.Draw(im);
font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', fsize);
text_intensity = 140
draw.text((195,150), "SIAM", (text_intensity,), font=font);
#draw.text((20,40), "read me?", (text_intensity,), font=font);
#im.show()
im.save(f'sample_N_256_{idx}.png');
np_im = np.asarray(im, dtype='float32')/255;

new_data = torch.from_numpy(np_im);
print(new_data.dtype)

fname_out = f"sample_{idx}_text"
torch.save(new_data, fname_out+'.pt');
savemat(fname_out+'.mat', {'data': new_data})




