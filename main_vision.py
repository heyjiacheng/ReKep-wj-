import os
import torch
import numpy as np
import argparse
import pyrealsense2 as rs

from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.utils import (
    bcolors,
    get_config,
)
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
        self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-small")
        self.intrinsics = self.load_camera_intrinsics()
        # sam2_model = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large")
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     model = sam2_model,
        #     points_per_side=64,
        #     points_per_batch=128,
        #     pred_iou_thresh=0.7,
        #     stability_score_thresh=0.92,
        #     stability_score_offset=0.7,
        #     crop_n_layers=1,
        #     box_nms_thresh=0.7,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=25.0,
        #     use_m2m=True,
        # )

    def load_camera_intrinsics(self):
        # Load the JSON file containing the camera calibration
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        intrinsics = depth_profile.get_intrinsics()

        pipeline.stop()

        return intrinsics, depth_scale

    def depth_to_pointcloud(self, depth):
        intrinsics, depth_scale = self.intrinsics

        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - intrinsics.ppx) / intrinsics.fx
        y = (v.flatten() - intrinsics.ppy) / intrinsics.fy

        z = depth.flatten() * depth_scale
        x = np.multiply(x, z)
        y = np.multiply(y, z)

    
        points = np.stack((x, y, z), axis = -1)
        return points    
    
    def perform_task(self, instruction, data_path, frame_number):
        # Load color and depth images
        color_path = os.path.join(data_path, f'color_{frame_number:06d}.npy')
        depth_path = os.path.join(data_path, f'depth_{frame_number:06d}.npy')

        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Color or depth frame not found for frame {frame_number}")

        rgb = np.load(color_path)
        depth = np.load(depth_path)

        print(f"Debug: Input image shape: {rgb.shape}") # (480, 640, 3)
        print(f"Debug: Input depth shape: {depth.shape}") # (480, 640)  

        # Generate masks
        masks_dict = self.mask_generator.generate(rgb)
        masks = [m['segmentation'] for m in masks_dict]
        print(f"Debug: Generated {len(masks)} masks")
        print(f"Debug: masks shape: {masks[0].shape}")
        print(f"Debug: Type of masks: {type(masks)}")
        # TODO: Add point cloud data from DepthPro model 
        # replace env.get_cam_obs() with Mujoco camera observation

        # Generate point cloud from depth image
        points = self.depth_to_pointcloud(depth)
        print(f"Debug: Generated point cloud with shape: {points.shape}")
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
        # Save the annotated image with keypoints
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('on')
        plt.title('Annotated Image with Keypoints')
        plt.savefig('data/rekep_with_keypoints.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Save the image with SAM2 masks
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        for mask in masks:
            plt.contour(mask['segmentation'], colors='r', linewidths=0.5, alpha=0.7)
        plt.axis('on')
        plt.title('Image with SAM2 Masks')
        plt.savefig('data/rekep_with_masks.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=True, help='Instruction for the task')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory containing color and depth frames')
    parser.add_argument('--frame_number', type=int, required=True, help='Frame number to process')
    parser.add_argument('--visualize', action='store_true', help='Visualize the keypoints on the image')
    args = parser.parse_args()

    main = MainVision(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, data_path=args.data_path, frame_number=args.frame_number)
