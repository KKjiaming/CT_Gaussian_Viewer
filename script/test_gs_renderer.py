"""Numerical checks of direct GPU projection; requires an EGL OpenGL context."""
import unittest

import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from gs_renderer import GaussianRenderer, camera_basis


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.ctx = moderngl.create_standalone_context(require=330, backend='egl')
        except Exception as error:
            self.skipTest(f'EGL unavailable: {error}')
        self.output = self.ctx.simple_framebuffer((129, 129), components=3, dtype='f4')
        self.model = dict(centers=np.zeros((1, 3)), scales=np.array([[.15, .3, .45]]),
                          rotations=Rotation.from_euler('xyz', [25, 40, 15], degrees=True).as_matrix()[None],
                          density=np.array([2.]))

    def tearDown(self):
        if hasattr(self, 'output'):
            self.output.release()
            self.ctx.release()

    def projection(self, model, minimum_density=0, maximum_ratio=1e9):
        renderer = GaussianRenderer(self.ctx, model)
        try:
            renderer.render((129, 129), self.output, np.zeros(3), .7, .4, 1, 1,
                            minimum_density, maximum_ratio, roll=.3)
            return np.frombuffer(renderer.fbo.read(components=1, dtype='f4'), dtype='f4').reshape(129,129)
        finally:
            renderer.release()

    def test_rotated_gaussian_against_numerical_ray_integral(self):
        image = self.projection(self.model)
        right, up = camera_basis(.7, .4, .3)
        direction = np.cross(right, up)
        rotation = self.model['rotations'][0]
        t = np.linspace(-4, 4, 20001)
        for row, column in ((64, 64), (70, 76), (55, 60)):
            origin = ((column+.5)/129*2-1)*right + ((row+.5)/129*2-1)*up
            local = (origin[None]+t[:,None]*direction) @ rotation / self.model['scales'][0]
            expected = np.trapz(2*np.exp(-.5*np.sum(local**2, axis=1)), t)
            self.assertAlmostEqual(float(image[row,column]), expected, delta=expected*2e-5)

    def test_overlaps_add_and_filters_remove(self):
        one = self.projection(self.model)
        two = self.projection({key: np.repeat(value, 2, axis=0) for key,value in self.model.items()})
        np.testing.assert_allclose(two, one*2, atol=1e-6)
        self.assertEqual(float(self.projection(self.model, minimum_density=3).max()), 0)
        self.assertEqual(float(self.projection(self.model, maximum_ratio=2).max()), 0)


if __name__ == '__main__':
    unittest.main()
