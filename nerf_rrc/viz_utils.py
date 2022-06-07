from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as p


class SphereMarker:
    def __init__(self, radius, position, color=(0, 1, 0, 0.5)):
        """
        Create a sphere marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
        """
        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        self._position = position
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def set_state(self, position):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
        """
        self._position = position
        orientation = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(self.body_id, position, orientation)

    @property
    def position(self):
        return self._position

    def __del__(self):
        """
        Removes the visual object from the environment
        """
        # At this point it may be that pybullet was already shut down. To avoid
        # an error, only remove the object if the simulation is still running.
        if p.isConnected():
            p.removeBody(self.body_id)


class VisualMarkers:
    """Visualize spheres on the specified points"""

    def __init__(self):
        self.markers = []

    def add(self, points, radius=0.015, color=None):
        if isinstance(points[0], (int, float)):
            points = [points]
        if color is None:
            color = (0, 1, 1, 0.5)
        for point in points:
            self.markers.append(SphereMarker(radius, point, color=color))

    def remove(self):
        self.markers = []

    def __del__(self):
        for marker in self.markers:
            marker.__del__()
