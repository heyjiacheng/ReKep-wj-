import yaml
import pyrealsense2 as rs

def initialize_realsense():
    # Load camera config
    with open('./configs/camera.yaml', 'r') as f:
        config = yaml.safe_load(f)['realsense']
    
    # Initialize pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    
    # Find D435i device
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.get_info(rs.camera_info.name).lower().find('d435i') >= 0:
            if config['serial_number']:  # If serial number specified
                if dev.get_info(rs.camera_info.serial_number) == config['serial_number']:
                    rs_config.enable_device(dev.get_info(rs.camera_info.serial_number))
            else:  # Use first D435i found
                rs_config.enable_device(dev.get_info(rs.camera_info.serial_number))
                break
    
    # Configure streams
    rs_config.enable_stream(rs.stream.depth, config['resolution']['width'], 
                          config['resolution']['height'], rs.format.z16, config['fps'])
    rs_config.enable_stream(rs.stream.color, config['resolution']['width'], 
                          config['resolution']['height'], rs.format.rgb8, config['fps'])
    
    return pipeline, rs_config