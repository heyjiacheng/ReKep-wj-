

import os
import numpy as np
from PIL import Image
import pdb
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load the image
img_path = '/home/tonyw/VLM/ReKep/data/pen.png'
image = Image.open(img_path)
image = np.array(image.convert("RGB"))

# Initialize the SAM2 predictor
# predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large")
# Set the image
# predictor.set_image(image)

# Generate masks for the entire image
# masks, scores, _ = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     multimask_output=True,
# )
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())
pdb.set_trace()
# Create output directory
img_name = os.path.splitext(os.path.basename(img_path))[0]
output_dir = f'./data/mask/{img_name}_sam2'
os.makedirs(output_dir, exist_ok=True)

# Convert masks to uint8 and stack them
masks_uint8 = np.stack([(mask > 0.5).astype(np.uint8) for mask in masks['segmentation'], axis=-1)

# Save masks as a single .npy file
masks_path = os.path.join(output_dir, f'{img_name}_masks.npy')
np.save(masks_path, masks_uint8)

print(f"Saved {masks_uint8.shape[-1]} masks to {masks_path}")