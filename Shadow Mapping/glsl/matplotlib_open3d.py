import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh("01.off")
    o3d.visualization.draw([mesh])