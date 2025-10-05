import dataclasses
import os

import moderngl as mgl
import numpy as np

@dataclasses.dataclass
class MilRendererConfig:
    width: int
    height: int
    fps: float
    input_path: str
    video_path: str

class MilRenderer:
    def __init__(self):
        try:
            self.ctx = mgl.create_context(standalone=True)
        except Exception as e:
            if "XOpenDisplay" in repr(e):
                self.ctx = mgl.create_context(standalone=True, backend="egl")
            else:
                raise RuntimeError("Failed to create rendering context") from e
        
        self._initialized = False
        
    def initialize(self, cfg: MilRendererConfig):
        if self._initialized:
            raise RuntimeError("Renderer already initialized")

        if not isinstance(cfg, MilRendererConfig):
            raise ValueError(f"Invalid config type: {type(cfg)}")
        
        try:
            cfg.width = float(cfg.width)
            cfg.height = float(cfg.height)

            if not cfg.width.is_integer() or not cfg.height.is_integer():
                raise Exception("size must be integer")
            
            cfg.width = int(cfg.width)
            cfg.height = int(cfg.height)

            if cfg.width <= 0 or cfg.height <= 0:
                raise Exception("size must be positive")
        except Exception as e:
            raise ValueError(f"Invalid resolution: {cfg.width}x{cfg.height}") from e
        
        try:
            cfg.fps = float(cfg.fps)
            
            if cfg.fps <= 0.0:
                raise Exception("fps must be positive")
        except Exception as e:
            raise ValueError(f"Invalid fps: {cfg.fps}") from e
        
        if not os.path.exists(cfg.input_path):
            raise ValueError(f"Invalid input path: {cfg.input_path} is not exists")
        
        if not os.path.isfile(cfg.input_path):
            raise ValueError(f"Invalid input path: {cfg.input_path} is not a file")
        
        if os.path.isdir(cfg.video_path):
            raise ValueError(f"Invalid video path: {cfg.video_path} is a directory")
        
        if os.path.exists(cfg.video_path):
            raise ValueError(f"Invalid video path: {cfg.video_path} is already exists")

        self.ctx.viewport = (0, 0, cfg.width, cfg.height)
        self._initialized = True
    
    def run(self):
        pass
    
if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--input", type=str, required=True)
    aparser.add_argument("-o", "--output", type=str, required=True)
    aparser.add_argument("-s-w", "--width", type=int, default=1920)
    aparser.add_argument("-s-h", "--height", type=int, default=1080)
    aparser.add_argument("-f", "--fps", type=float, default=60.0)

    args = aparser.parse_args()
    renderer = MilRenderer()
    
    renderer.initialize(MilRendererConfig(
        width = args.width,
        height = args.height,
        fps = args.fps,
        input_path = args.input,
        video_path = args.output
    ))

    renderer.run()
