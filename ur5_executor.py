import json
import time
import asyncio
import os
import numpy as np
from ur_env.vacuum_gripper import VacuumGripper
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from ur_env.rotations import rotvec_2_quat, quat_2_rotvec, pose2rotvec, pose2quat


class UR5Executor:
    def __init__(self, ip_address, action_file="./outputs/action.json"):
        self.ip_address = ip_address
        self.action_file = action_file
        self.rtde_control = None
        self.rtde_receive = None
        self.current_gripper_state = None  # To track gripper state (1.0=closed, 0.0=open)
        
    def connect(self):
        """Connect to the UR5 robot"""
        try:
            self.rtde_control = RTDEControlInterface(self.ip_address)
            self.rtde_receive = RTDEReceiveInterface(self.ip_address)
            print(f"Successfully connected to UR5 at {self.ip_address}")
            return True
        except Exception as e:
            print(f"Error connecting to UR5: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from the UR5 robot"""
        if self.rtde_control:
            self.rtde_control.stopScript()
            print("Disconnected from UR5")
            
    def get_current_pose(self):
        """Get the current pose of the robot"""
        if self.rtde_receive:
            task_space_pose = self.rtde_receive.getActualTCPPose()
            # Convert rotation vector to degrees for display
            rot_degrees = [np.degrees(angle) for angle in task_space_pose[3:]]
            print(f"Current pose: position {task_space_pose[:3]}, rotation {rot_degrees} (degrees)")
            return task_space_pose
        return None
        
    async def control_gripper(self, close=True):
        """Control the vacuum gripper"""
        try:
            gripper = VacuumGripper(self.ip_address)
            await gripper.connect()
            await gripper.activate()
            
            if close:
                await gripper.close_gripper(force=100, speed=30)
                self.current_gripper_state = 1.0
                print("Gripper closed")
            else:
                await gripper.open_gripper(force=100, speed=30)
                self.current_gripper_state = 0.0
                print("Gripper opened")
                
            await gripper.disconnect()
            return True
        except Exception as e:
            print(f"Error controlling gripper: {e}")
            return False
            
    def execute_gripper_command(self, gripper_value):
        """Execute gripper command based on the gripper value"""
        # Only execute if there's a change in gripper state
        close_gripper = (gripper_value > 0.5)
        if self.current_gripper_state is None or close_gripper != (self.current_gripper_state > 0.5):
            try:
                # Get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Execute gripper command
                if loop.is_running():
                    asyncio.ensure_future(self.control_gripper(close=close_gripper))
                    # Give time for the command to execute
                    time.sleep(1.0)
                else:
                    loop.run_until_complete(self.control_gripper(close=close_gripper))
            except Exception as e:
                print(f"Error executing gripper command: {e}")
                
    def send_pose(self, pose, speed=0.05, acceleration=0.05):
        """Send a pose to the robot"""
        if self.rtde_control:
            try:
                # Convert Euler angles from degrees to radians for moveL command
                pose_radians = pose[:3] + [np.radians(angle) for angle in pose[3:6]]
                # Apply corrections to match RTDEControl expectations
                pose_corrected = pose_radians[:3] + [-val for val in pose_radians[3:]]
                
                # Execute the movement
                self.rtde_control.moveL(pose_corrected, speed, acceleration)
                print(f"Moved to pose: {pose[:3]}, {pose[3:6]}")
                return True
            except Exception as e:
                print(f"Error sending pose to robot: {e}")
                return False
        return False
        
    def load_action_sequence(self):
        """Load action sequence from JSON file"""
        try:
            if not os.path.exists(self.action_file):
                print(f"Action file not found: {self.action_file}")
                return None
                
            with open(self.action_file, 'r') as f:
                data = json.load(f)
                
            if 'ee_action_seq' not in data:
                print(f"Invalid action file format: missing 'ee_action_seq'")
                return None
                
            print(f"Loaded {len(data['ee_action_seq'])} actions from {self.action_file}")
            return data
        except Exception as e:
            print(f"Error loading action sequence: {e}")
            return None
            
    def execute_sequence(self, speed=0.05, acceleration=0.05, delay=1.0, return_to_start=True, position_only=False):
        """Execute the full action sequence
        
        Args:
            speed: Movement speed 
            acceleration: Movement acceleration
            delay: Delay between movements
            return_to_start: Whether to return to starting position after execution
            position_only: If True, only execute position changes but keep current orientation
        """
        # Load the action sequence
        data = self.load_action_sequence()
        if not data:
            return False
            
        action_seq = data['ee_action_seq']
        stage = data.get('stage', 1)
        print(f"Executing stage {stage} with {len(action_seq)} actions")
        
        # Connect to the robot
        if not self.connect():
            return False
            
        try:
            # Get current pose - store as starting pose
            starting_pose = self.get_current_pose()
            if starting_pose is None:
                print("Failed to get current pose")
                return False
                
            # Get current orientation in degrees for position_only mode
            current_orientation = None
            if position_only:
                current_orientation = [np.degrees(angle) for angle in starting_pose[3:]]
                print(f"Position-only mode: Maintaining current orientation: {current_orientation}")
                
            # Initialize gripper state based on current state
            self.current_gripper_state = 0.0  # Assume open initially
            
            # Execute each action in sequence
            for i, action in enumerate(action_seq):
                print(f"\nExecuting action {i+1}/{len(action_seq)}")
                
                # Extract position, orientation, and gripper command
                position = action[:3]
                
                # Use original orientation or keep current orientation if position_only is True
                orientation = current_orientation if position_only else action[3:6]
                
                gripper_cmd = action[6]
                
                # Create the pose
                pose = position + orientation
                
                # Execute the pose
                if not self.send_pose(pose, speed, acceleration):
                    print(f"Failed to execute pose {i+1}, aborting sequence")
                    return False
                    
                # Wait a bit between movements
                time.sleep(delay)
                
                # Execute gripper action if it's the last action or if there's a change
                if i == len(action_seq) - 1 or action_seq[i+1][6] != gripper_cmd:
                    self.execute_gripper_command(gripper_cmd)
                    time.sleep(delay)
                    
            print("\nAction sequence completed successfully")
            
            # Return to starting position if requested
            if return_to_start:
                print("\nReturning to starting position...")
                
                # First ensure gripper is open
                if self.current_gripper_state > 0.5:  # If gripper is closed
                    self.execute_gripper_command(0.0)  # Open the gripper
                    time.sleep(delay)
                
                # Convert the starting pose to the format expected by send_pose
                starting_pos = starting_pose[:3]
                starting_rot = [np.degrees(angle) for angle in starting_pose[3:]]
                starting_pose_for_send = starting_pos + starting_rot
                
                # Move back to starting position
                if self.send_pose(starting_pose_for_send, speed, acceleration):
                    print("Successfully returned to starting position")
                else:
                    print("Failed to return to starting position")
            
            return True
            
        except Exception as e:
            print(f"Error executing action sequence: {e}")
            return False
            
        finally:
            # Always disconnect from the robot when done
            self.disconnect()


if __name__ == "__main__":
    # UR5 robot IP address
    UR5_IP = "192.168.1.60"
    
    # Path to action file
    ACTION_FILE = "./outputs/action.json"
    
    # Create executor and run the sequence
    executor = UR5Executor(UR5_IP, ACTION_FILE)
    
    try:
        # Example: Execute with position-only mode (keep current orientation)
        # Set position_only=True to maintain the current orientation during movements
        
        success = executor.execute_sequence(
            speed=0.05, 
            acceleration=0.05, 
            delay=1.0, 
            return_to_start=True,
            position_only=True  # Only execute position changes, maintain orientation
        )
        
        if success:
            print("Successfully executed all actions")
        else:
            print("Failed to execute action sequence")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        executor.disconnect() 