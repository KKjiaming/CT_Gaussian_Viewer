"""Run with CT_VIEWER_UI_TESTS=1 and a working VTK/EGL graphics backend."""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis_viewer import AnalysisViewer, reference_slice


class ReferenceAlignmentTests(unittest.TestCase):
    def test_native_voxel_slice_matches_comparison_transpose(self):
        shape = (16, 16, 16)
        volume = np.arange(np.prod(shape), dtype="f4").reshape(shape)
        bbox = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        for axis in range(3):
            index = shape[axis] // 2
            coordinate = bbox[0, axis] + (index + 0.5) / shape[axis] * (
                bbox[1, axis] - bbox[0, axis]
            )
            actual = reference_slice(volume, axis, coordinate, bbox, shape[axis])
            np.testing.assert_array_equal(actual, np.take(volume, index, axis=axis).T)


@unittest.skipUnless(os.environ.get("CT_VIEWER_UI_TESTS") == "1", "Opt-in VTK/EGL integration test")
class KeyboardIntegrationTests(unittest.TestCase):
    def test_native_key_events_preserve_panels_camera_and_slice_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shape = (16, 16, 16)
            (root / "meta.json").write_text(
                json.dumps(dict(bbox=[[-1, -1, -1], [1, 1, 1]], scanner=dict(nVoxel=shape)))
            )
            np.save(root / "volume.npy", np.ones(shape, dtype="f4") * 0.5)
            parameters = dict(
                centers=np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]]),
                scales=np.ones((2, 3)) * 0.2,
                rotations=np.repeat(np.eye(3)[None], 2, axis=0),
                density=np.ones(2),
            )
            args = SimpleNamespace(
                metadata=root / "meta.json",
                analysis_resolution=16,
                fact_volume=root / "volume.npy",
                gt_volume=None,
                r2_volume=None,
                method="fact_gs",
                min_density=0.05,
                max_ratio=1000,
                max_count=100,
                gray_range=None,
                exposure=2.0,
                camera_save=None,
                output=root,
                camera=None,
                candidate_count=8,
                gaussian_style="solid",
            )
            viewer = AnalysisViewer(args, parameters, root / "parameters.npz", {})
            try:
                viewer.plotter.show(auto_close=False, interactive=False)
                snapshots = [viewer.images[(1, i)].cell_data["rgb"].copy() for i in range(3)]
                np.testing.assert_array_equal(viewer.images[(0, 1)].cell_data["rgb"], snapshots[0])
                viewports = [r.GetViewport() for r in viewer.plotter.renderers]
                cameras = [
                    (tuple(r.camera.position), tuple(r.camera.up), r.camera.parallel_scale)
                    for r in list(viewer.plotter.renderers)[1:]
                ]
                interactor = viewer.plotter.iren.interactor

                def key(value):
                    interactor.SetEventInformation(
                        1500, 100, 0, int(value.isupper()), value[0], 0, value
                    )
                    interactor.KeyPressEvent()
                    interactor.CharEvent()
                    interactor.KeyReleaseEvent()

                for value in ("x", "y", "z", "X", "Y", "Z", "1", "2", "3", "2", "1"):
                    key(value)
                    self.assertFalse(viewer.plotter.ren_win.GetStereoRender())
                    self.assertEqual([r.GetViewport() for r in viewer.plotter.renderers], viewports)
                    self.assertTrue(all(r.GetDraw() for r in viewer.plotter.renderers))
                    for i in range(3):
                        np.testing.assert_array_equal(
                            viewer.images[(1, i)].cell_data["rgb"], snapshots[i]
                        )
                    actual = [
                        (tuple(r.camera.position), tuple(r.camera.up), r.camera.parallel_scale)
                        for r in list(viewer.plotter.renderers)[1:]
                    ]
                    self.assertEqual(actual, cameras)
                    visible = [
                        viewer.plotter.renderers[0].actors[f"plane_{i}"].visibility
                        for i in range(3)
                    ]
                    self.assertEqual(sum(visible), 1)
                    self.assertTrue(visible[viewer.aligned_axis])
                    np.testing.assert_array_equal(
                        viewer.images[(0, 1)].cell_data["rgb"], snapshots[viewer.aligned_axis]
                    )
                key("v")
                self.assertEqual(viewer.slice_source, 1)  # exactly one callback, no isometric reset
                self.assertEqual(viewer.gray_range, [0.0, 1.0])
                viewer.pick_slice(0, (0.5, 0.5, 0))
                self.assertIsNotNone(viewer.model.selected)
                before = viewer.slice_current[0].copy()
                gate = threading.Event()
                compute = viewer.compute_edit

                def slow_compute(*args):
                    gate.wait(timeout=2)
                    return compute(*args)

                viewer.compute_edit = slow_compute
                started = time.monotonic()
                key("h")
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertIsNotNone(viewer.edit_future)
                self.assertTrue(viewer.busy)
                key("w")  # GUI callbacks continue while the edit worker is deliberately blocked
                gate.set()

                def wait_edit():
                    deadline = time.monotonic() + 10
                    while viewer.edit_future is not None and time.monotonic() < deadline:
                        interactor.TimerEvent()
                        time.sleep(0.01)
                    self.assertIsNone(viewer.edit_future)
                    self.assertFalse(viewer.busy)

                wait_edit()
                self.assertTrue(np.any(viewer.slice_current[0] != before))
                from analysis_viewer import display_rgb

                edited_rgb = display_rgb([viewer.slice_current[0]], *viewer.gray_range).reshape(
                    -1, 3
                )
                np.testing.assert_array_equal(viewer.images[(0, 1)].cell_data["rgb"], edited_rgb)
                key("t")
                original_rgb = display_rgb([viewer.slice_baseline[0]], *viewer.gray_range).reshape(
                    -1, 3
                )
                np.testing.assert_array_equal(viewer.images[(0, 1)].cell_data["rgb"], original_rgb)
                key("t")
                self.assertEqual(viewer.comparison_source, 2)
                self.assertTrue(np.any(viewer.images[(0, 1)].cell_data["rgb"] > 0))
                key("t")
                np.testing.assert_array_equal(viewer.images[(0, 1)].cell_data["rgb"], edited_rgb)
                key("u")
                wait_edit()
                np.testing.assert_allclose(viewer.slice_current[0], before, atol=1e-6)
            finally:
                viewer.edit_executor.shutdown(wait=True, cancel_futures=True)
                viewer.gpu.release()
                viewer.plotter.close()


if __name__ == "__main__":
    unittest.main()
