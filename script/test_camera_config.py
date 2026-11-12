import tempfile
import unittest
from pathlib import Path

import numpy as np

from camera_config import validate_camera, load_camera, save_camera, aligned_camera
from gs_renderer import camera_basis, configure_camera


class CameraTests(unittest.TestCase):
    def test_axis_views_match_slice_screen_axes_and_reverse(self):
        bbox = np.array([[-1,-2,-3],[1,2,3]])
        slices = [.1,.2,.3]
        for axis in range(3):
            others = [i for i in range(3) if i!=axis]
            for reverse in (False,True):
                camera = aligned_camera(axis,reverse,bbox,slices,(600,500))
                direction = np.array(camera['position'])-camera['target']
                direction /= np.linalg.norm(direction)
                screen_right = np.cross(camera['up'],direction)
                np.testing.assert_allclose(screen_right,np.eye(3)[others[0]]*(-1 if reverse else 1))
                np.testing.assert_allclose(camera['up'],np.eye(3)[others[1]])
                self.assertAlmostEqual(camera['target'][axis],slices[axis])
                self.assertGreater(camera['parallel_scale'],0)

    def test_world_camera_round_trip_preserves_direction_and_roll(self):
        for position in ([3, 2, 1], [0, 0, 3], [-2, 1, -3]):
            config = dict(position=position, target=[.1, -.2, .3], up=[0, 1, .4],
                          parallel_scale=1.25, projection='orthographic')
            state = dict(yaw=0., pitch=0., span=1., distance=3.)
            target = np.zeros(3)
            configure_camera(config, state, target)
            right, up = camera_basis(state['yaw'], state['pitch'], state['roll'])
            direction = np.cross(right, up)
            np.testing.assert_allclose(target+direction*state['distance'], position, atol=1e-12)
            desired_up = np.array(config['up'])
            desired_up -= np.dot(desired_up, direction)*direction
            desired_up /= np.linalg.norm(desired_up)
            np.testing.assert_allclose(up, desired_up, atol=1e-12)
            self.assertEqual(state['span'], 1.25)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'camera.json'
                save_camera(path, config)
                self.assertEqual(load_camera(path), validate_camera(config))

    def test_invalid_camera_rejected(self):
        for config in ({'parallel_scale': 0}, {'up': [0, 0, 0]}, {'position': [float('nan'), 0, 1]},
                       {'position': [0,0,0], 'target': [0,0,0]}, {'clipping_range': [10,1]},
                       {'window_size': [0,900]}, {'view_angle': 180}, {'unknown': 1},
                       {'position': [0,0,1], 'target': [0,0,0], 'up': [0,0,2]}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_camera(config)


if __name__ == '__main__':
    unittest.main()
