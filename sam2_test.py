import torch
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

img_path = '/home/tonyw/VLM/ReKep/data/mask_pen_0.png'
image = Image.open(img_path)
image_np = np.array(image.convert("RGB"))

# 打印图像信息~/VLM/ReKep
print(f"Image shape: {image_np.shape}")
print(f"Image dtype: {image_np.dtype}")
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_np)
    masks, _, _ = predictor.predict('pot')
    print(masks)