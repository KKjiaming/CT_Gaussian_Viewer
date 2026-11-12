import unittest
import tempfile
from pathlib import Path

import moderngl
import numpy as np
from scipy.spatial.transform import Rotation

from gaussian_analysis import (AnalysisGPU, EditableGaussians, point_contributions,
                               region_contributions, pixel_world, rank_contributions,slice_gaussian_ids)


def model():
    return dict(centers=np.array([[.1,-.2,.05],[-.1,.3,-.05],[.25,.1,.2]]),
                scales=np.array([[.2,.35,.4],[.25,.2,.3],[.15,.3,.2]]),
                rotations=Rotation.from_euler('xyz',[[25,40,15],[60,5,20],[0,20,50]],degrees=True).as_matrix(),
                density=np.array([2.,1.,.5]))


class EditingTests(unittest.TestCase):
    def test_slice_membership_uses_rotated_support_and_moves_with_plane(self):
        p=dict(centers=np.array([[.2,0,0],[.2,0,0],[.8,0,0]]),
               scales=np.array([[.01,.01,1.],[.01,.01,1.],[.01,.01,.01]]),
               rotations=np.stack([Rotation.from_euler('y',90,degrees=True).as_matrix(),np.eye(3),np.eye(3)]),
               density=np.ones(3))
        hidden=np.zeros(3,dtype=bool)
        ids,_=slice_gaussian_ids(p,0,0.,hidden)
        np.testing.assert_array_equal(ids,[0])
        ids,_=slice_gaussian_ids(p,0,.8,hidden)
        np.testing.assert_array_equal(ids,[0,2])
        hidden[0]=True
        ids,_=slice_gaussian_ids(p,0,.8,hidden)
        np.testing.assert_array_equal(ids,[2])

    def test_export_reimport_preserves_original_ids(self):
        from density_viewer import load_parameters
        p = model()
        p['original_ids'] = np.array([100,250,300])
        editor = EditableGaussians(p)
        editor.select(1)
        editor.hide_selected()
        with tempfile.TemporaryDirectory() as folder:
            editor.export(folder,'source.npz')
            loaded = load_parameters(Path(folder)/'edited_parameters.npz')
            np.testing.assert_array_equal(loaded['original_ids'],[100,300])
            reimported = EditableGaussians(loaded)
            reimported.select(1)
            self.assertEqual(reimported.selected_parameters()['original_id'],300)

    def test_hide_group_undo_and_source_immutability(self):
        p = model()
        before = p['density'].copy()
        editor = EditableGaussians(p)
        editor.select(2)
        self.assertTrue(editor.hide_selected())
        self.assertFalse(editor.hide_selected())
        self.assertEqual(editor.selected_parameters()['original_id'],2)
        editor.hide_ids([0,1])
        self.assertEqual(float(editor.density.sum()),0)
        editor.undo()
        np.testing.assert_array_equal(editor.hidden,[False,False,True])
        editor.undo()
        np.testing.assert_array_equal(editor.density,before)
        np.testing.assert_array_equal(p['density'],before)

    def test_actual_contribution_ranking_is_not_distance_ranking(self):
        p = dict(centers=np.array([[0.,0.,0.],[.2,0.,0.]]),
                 scales=np.ones((2,3)),rotations=np.repeat(np.eye(3)[None],2,axis=0),
                 density=np.array([.01,2.]))
        values = point_contributions(p,[0,0,0])
        ids,_,fractions,total = rank_contributions(values,1)
        self.assertEqual(ids[0],1)
        self.assertAlmostEqual(fractions[0],values[1]/values.sum())
        self.assertGreater(total,values[1])
        np.testing.assert_array_equal(point_contributions(p,[0,0,0],[False,True]),[.01,0])


class SliceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.ctx = moderngl.create_standalone_context(require=330,backend='egl')
        except Exception as error:
            self.skipTest(f'EGL unavailable: {error}')
        self.p = model()
        self.bbox = np.array([[-1.5,-1.2,-1.],[1.5,1.2,1.]])
        self.gpu = AnalysisGPU(self.ctx,self.p,self.bbox)

    def tearDown(self):
        if hasattr(self,'gpu'):
            self.gpu.release()
            self.ctx.release()

    def test_all_planes_match_density_and_roi_scores(self):
        resolution = 65
        for axis in range(3):
            image = self.gpu.slice(axis,.1,resolution)
            for row,col in ((32,32),(25,35),(40,29)):
                world = pixel_world(axis,row,col,.1,self.bbox,resolution)
                expected = point_contributions(self.p,world).sum()
                self.assertAlmostEqual(float(image[row,col]),expected,delta=2e-5)
            roi = (20,42,22,40)
            scores = region_contributions(self.p,axis,.1,self.bbox,resolution,roi)
            self.assertAlmostEqual(scores.sum(),float(image[20:43,22:41].sum()),delta=.002)

    def test_hidden_density_updates_slice_and_projection_then_restores(self):
        before = self.gpu.slice(2,.1,65)
        camera = dict(position=[3,2,1],target=[0,0,0],up=[0,0,1])
        projection_before = self.gpu.projection(camera,65)
        edited = self.p['density'].copy()
        edited[0] = 0
        self.gpu.update_density(edited)
        after = self.gpu.slice(2,.1,65)
        projection_after = self.gpu.projection(camera,65)
        self.assertGreater(float((before-after).max()),0)
        self.assertGreater(float((projection_before-projection_after).max()),0)
        remaining = {k:v[1:] for k,v in self.p.items()}
        world = pixel_world(2,32,32,.1,self.bbox,65)
        self.assertAlmostEqual(float(after[32,32]),point_contributions(remaining,world).sum(),delta=2e-5)
        self.gpu.update_density(self.p['density'])
        np.testing.assert_allclose(self.gpu.slice(2,.1,65),before,atol=1e-6)
        np.testing.assert_allclose(self.gpu.projection(camera,65),projection_before,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
