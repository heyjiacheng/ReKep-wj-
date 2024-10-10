import torch
import numpy as np
import argparse
import cv2
import os
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from utils import (
    bcolors,
    get_config,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class MainVision:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.visualize = visualize
        # Set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # Initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])

    def perform_task(self, instruction, image_path, mask_path=None):
        # Load image
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        print(f"Debug: Input image shape: {rgb.shape}")

        # Load mask if provided
        if mask_path:
            try:
                masks = np.load(mask_path)
                if masks is None:
                    raise FileNotFoundError(f"Mask not found at {mask_path}")
                print(f"Debug: Input masks shape: {masks.shape}")
                
                # Ensure masks are in the correct format (uint8)
                masks = masks.astype(np.uint8)
                
                # If masks is 2D, expand to 3D
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, axis=-1)
                
                print(f"Debug: Processed masks shape: {masks.shape}")
            except Exception as e:
                print(f"Error loading mask: {e}")
                masks = None
        else:
            masks = None
            print("Debug: No mask provided")

        # Since we don't have point cloud data, we can simulate or omit it
        # For now, we'll set points to None
        points = None
        # TODO: Add point cloud data from RGBD camera
          # simulate this from omnigibson
          # replace env.get_cam_obs() with Mujoco camera observation
        
        # Question: simulator or video is fine?
        # Answer: 
        # Question: how to turn omnigibson into mujoco?
        # Answer: 
        # it seems working on the code is more satisfying than just reading the paper
        # i should reimplement the part in my own way, as the professor said it is possible
        # since i am hired by Jason, then i should have ability to solve it independently

        # mujuco  √
          
        # dataset 
        # simulate from Image, Depth estimation 

        # ====================================
        # = Keypoint Proposal and Constraint Generation
        # ====================================
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
        if self.visualize:
            self._show_image(projected_img)
        metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
        print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')

    def _show_image(self, image):
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=True, help='Instruction for the task')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input RGB image')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the segmentation mask')
    parser.add_argument('--visualize', action='store_true', help='Visualize the keypoints on the image')
    args = parser.parse_args()

    main = MainVision(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, image_path=args.image_path, mask_path=args.mask_path)
