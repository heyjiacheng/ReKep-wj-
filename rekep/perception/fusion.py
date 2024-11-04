# fusion and registration for depth and rgb
# point cloud calibration
class MultiViewFusion:
    def __init__(self, config):
        self.zed = ZED2Camera(config)
        self.depth_pro = DepthProCamera(config)
        
    def fuse_point_clouds(self, frames):
        """Fuse multiple view point clouds"""
        point_clouds = []
        for frame in frames:
            pc = self.process_frame(frame)
            point_clouds.append(pc)
        return self.register_point_clouds(point_clouds)