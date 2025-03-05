import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import argparse
import json
import os

def load_camera_extrinsics(extrinsics_path):
    """Load camera extrinsics from YAML file"""
    with open(extrinsics_path, 'r') as f:
        extrinsics_data = yaml.safe_load(f)
    
    # Extract transformation parameters
    qw = extrinsics_data['transformation']['qw']
    qx = extrinsics_data['transformation']['qx']
    qy = extrinsics_data['transformation']['qy']
    qz = extrinsics_data['transformation']['qz']
    tx = extrinsics_data['transformation']['x']
    ty = extrinsics_data['transformation']['y']
    tz = extrinsics_data['transformation']['z']
    
    # Create rotation matrix from quaternion
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # Create 4x4 transformation matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot
    extrinsics[:3, 3] = [tx, ty, tz]
    
    return extrinsics

def draw_coordinate_frame(ax, transform, scale=0.05, label=None):
    """Draw a coordinate frame based on a transformation matrix"""
    origin = transform[:3, 3]
    
    # X, Y, Z axes
    x_axis = origin + scale * transform[:3, 0]
    y_axis = origin + scale * transform[:3, 1]
    z_axis = origin + scale * transform[:3, 2]
    
    # Draw axes
    ax.quiver(*origin, *(x_axis - origin), color='r', label='X' if label else None)
    ax.quiver(*origin, *(y_axis - origin), color='g', label='Y' if label else None)
    ax.quiver(*origin, *(z_axis - origin), color='b', label='Z' if label else None)
    
    if label:
        ax.text(*origin, label, fontsize=12)

def transform_keypoints_to_world(keypoints, ee_pose, ee2camera):
    """
    Transform keypoints from camera coordinate system to world coordinate system
    """
    # Convert to numpy array
    keypoints = np.array(keypoints)
    
    # Convert to homogeneous coordinates
    keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
    
    # EE frame with handedness correction
    position = ee_pose[:3]
    quat = np.array([ee_pose[4], ee_pose[5], ee_pose[6], ee_pose[3]])  # [qx,qy,qz,qw]
    rotation = R.from_quat(quat).as_matrix()
    
    # Apply handedness correction - reverse X and Z axes
    rot_correct = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    rotation_corrected = rotation @ rot_correct
    
    base2ee = np.eye(4)
    base2ee[:3, :3] = rotation_corrected
    base2ee[:3, 3] = position
    
    # Camera frame
    camera_frame_incorrect = base2ee @ ee2camera
    
    # Create camera axes correction matrix
    camera_axes_correction = np.array([
        [0, 0, 1],  # New x-axis is old z-axis
        [-1, 0, 0], # New y-axis is negative old x-axis
        [0, -1, 0]  # New z-axis is negative old y-axis
    ])
    
    # Apply the correction to the camera frame rotation part
    camera_frame = camera_frame_incorrect.copy()
    camera_frame[:3, :3] = camera_frame_incorrect[:3, :3] @ camera_axes_correction
    
    # Apply transformation
    base_coords_homogeneous = (camera_frame @ keypoints_homogeneous.T).T
    
    # Convert back to non-homogeneous coordinates
    base_coords = base_coords_homogeneous[:, :3] / base_coords_homogeneous[:, 3, np.newaxis]
    
    return base_coords

def load_keypoints(rekep_program_dir):
    """Load keypoints from metadata.json"""
    metadata_path = os.path.join(rekep_program_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return []
    
    with open(metadata_path, 'r') as f:
        program_info = json.load(f)
    
    return program_info.get('init_keypoint_positions', [])

def visualize_frames(rekep_program_dir=None):
    """Visualize the base, EE, and camera coordinate frames with the correct interpretation"""
    # Load camera extrinsics
    extrinsics_path = '/home/xu/.ros/easy_handeye/easy_handeye_eye_on_hand.yaml'
    ee2camera = load_camera_extrinsics(extrinsics_path)
    
    # Get test EE pose
    ee_pose = np.array([-0.400482217, -0.127933676, 0.458474398, 
                         -0.0273204457, -0.0000377962955, -0.000260592389, 0.999626692])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Base frame is identity
    base_frame = np.eye(4)
    
    # EE frame with handedness correction
    position = ee_pose[:3]
    quat = np.array([ee_pose[4], ee_pose[5], ee_pose[6], ee_pose[3]])  # [qx,qy,qz,qw]
    rotation = R.from_quat(quat).as_matrix()
    
    # Apply handedness correction - reverse X and Z axes
    rot_correct = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    rotation_corrected = rotation @ rot_correct
    
    base2ee = np.eye(4)
    base2ee[:3, :3] = rotation_corrected
    base2ee[:3, 3] = position
    
    # Camera frame
    camera_frame_incorrect = base2ee @ ee2camera
    
    # Create camera axes correction matrix
    # This transforms camera axes according to:
    # - camera_x becomes camera_z
    # - camera_y becomes -camera_x
    # - camera_z becomes -camera_y
    camera_axes_correction = np.array([
        [0, 0, 1],  # New x-axis is old z-axis
        [-1, 0, 0], # New y-axis is negative old x-axis
        [0, -1, 0]  # New z-axis is negative old y-axis
    ])
    
    # Apply the correction to the camera frame rotation part
    camera_frame = camera_frame_incorrect.copy()
    camera_frame[:3, :3] = camera_frame_incorrect[:3, :3] @ camera_axes_correction
    
    # Draw coordinate frames
    draw_coordinate_frame(ax, base_frame, scale=0.1, label='Base')
    draw_coordinate_frame(ax, base2ee, scale=0.08, label='EE')
    draw_coordinate_frame(ax, camera_frame, scale=0.05, label='Camera')
    
    # Load and transform keypoints if rekep_program_dir is provided
    if rekep_program_dir:
        keypoints_camera = load_keypoints(rekep_program_dir)
        if keypoints_camera:
            print(f"Loaded {len(keypoints_camera)} keypoints from {rekep_program_dir}")
            print(f"Camera keypoints: {keypoints_camera}")
            
            # Transform keypoints to world coordinates
            keypoints_world = transform_keypoints_to_world(keypoints_camera, ee_pose, ee2camera)
            print(f"World keypoints: {keypoints_world}")
            
            # Plot keypoints in camera frame (as small spheres)
            keypoints_camera_homogeneous = np.hstack((keypoints_camera, np.ones((len(keypoints_camera), 1))))
            keypoints_camera_in_world = (camera_frame @ keypoints_camera_homogeneous.T).T
            keypoints_camera_in_world = keypoints_camera_in_world[:, :3] / keypoints_camera_in_world[:, 3, np.newaxis]
            
            ax.scatter(keypoints_camera_in_world[:, 0], keypoints_camera_in_world[:, 1], 
                      keypoints_camera_in_world[:, 2], color='purple', s=100, label='Keypoints (Camera Frame)')
            
            # Plot transformed keypoints in world frame
            ax.scatter(keypoints_world[:, 0], keypoints_world[:, 1], keypoints_world[:, 2], 
                      color='orange', s=100, label='Keypoints (World Frame)')
            
            # Add keypoint indices as text
            for i, (kp_cam, kp_world) in enumerate(zip(keypoints_camera_in_world, keypoints_world)):
                ax.text(kp_cam[0], kp_cam[1], kp_cam[2], f"{i}", color='purple', fontsize=12)
                ax.text(kp_world[0], kp_world[1], kp_world[2], f"{i}", color='orange', fontsize=12)
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set limits appropriately
    all_points = np.vstack([
        base_frame[:3, 3],
        base2ee[:3, 3],
        camera_frame[:3, 3]
    ])
    
    # Include keypoints in limits calculation if available
    if rekep_program_dir and 'keypoints_world' in locals():
        all_points = np.vstack([all_points, keypoints_world])
    
    min_vals = np.min(all_points, axis=0) - 0.1
    max_vals = np.max(all_points, axis=0) + 0.1
    
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    
    ax.legend()
    
    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    plt.title('Visualization of Base, EE, Camera Frames and Keypoints')
    plt.tight_layout()
    plt.savefig('frames_keypoints_visualization.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize coordinate frames and keypoints')
    parser.add_argument('--extrinsics', type=str, default='/home/xu/.ros/easy_handeye/easy_handeye_eye_on_hand.yaml',
                      help='Path to camera extrinsics YAML file')
    parser.add_argument('--rekep_dir', type=str, help='Path to ReKep program directory containing metadata.json')
    args = parser.parse_args()
    
    # If rekep_dir is not provided, try to find the most recent directory
    if not args.rekep_dir:
        vlm_query_dir = "/home/xu/Desktop/workspace/ReKep-wj-/vlm_query/"
        if os.path.exists(vlm_query_dir):
            vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                        if os.path.isdir(os.path.join(vlm_query_dir, d))]
            if vlm_dirs:
                args.rekep_dir = max(vlm_dirs, key=os.path.getmtime)
                print(f"\033[92mUsing most recent directory: {args.rekep_dir}\033[0m")
    
    visualize_frames(args.rekep_dir)

if __name__ == "__main__":
    main()