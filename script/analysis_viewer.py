"""Linked Gaussian picking, direct projection, editable slices and fixed references."""

from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile
from scipy.ndimage import map_coordinates

from camera_config import apply_pyvista_camera, capture_pyvista_camera, load_camera, save_camera
from gaussian_analysis import (
    EditableGaussians,
    pixel_world,
    point_contributions,
    rank_contributions,
    region_contributions,
    slice_gaussian_ids,
)


def load_reference(path, shape):
    """Load a finite XYZ NPY/TIFF volume matching the metadata shape.

    path is a local filename; shape is (nx, ny, nz), with no axis permutation.
    Returns float32 values, dividing uint8 previews by 255 only. Other input
    types keep their original scale. Volumes must already be spatially aligned.
    """
    path = Path(path)
    data = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else tifffile.imread(path)
    )
    if data.ndim != 3 or tuple(data.shape) != tuple(shape) or not np.isfinite(data).all():
        raise ValueError(
            f"{path}: expected finite XYZ volume with shape {tuple(shape)}; "
            "references must already share the metadata coordinate system"
        )
    return data.astype("f4") / (255 if data.dtype == np.uint8 else 1)


def reference_slice(volume, axis, coordinate, bbox, resolution):
    """Resample an XYZ reference at world-axis pixel centers using trilinear sampling.

    Args:
        volume: (nx, ny, nz) array already aligned with bbox.
        axis: Normal axis, 0=X, 1=Y, or 2=Z.
        coordinate: World position along the normal.
        bbox: (2, 3) lower/upper world XYZ bounds at voxel outer edges.
        resolution: Output pixel count along each square image dimension.

    Returns:
        (resolution, resolution) values with columns/rows increasing along the
        lower/higher remaining axes. Row zero is the lower world edge. Samples
        beyond the input's voxel-center range use nearest-edge extrapolation.
    """
    remaining = [i for i in range(3) if i != axis]
    u, v = np.meshgrid(
        *(
            bbox[0, i] + (np.arange(resolution) + 0.5) / resolution * (bbox[1, i] - bbox[0, i])
            for i in remaining
        ),
        indexing="xy",
    )
    points = np.empty((3, resolution, resolution), dtype=float)
    points[axis] = coordinate
    points[remaining[0]], points[remaining[1]] = u, v
    indices = (points - bbox[0, :, None, None]) / (bbox[1] - bbox[0])[:, None, None]
    indices = indices * np.asarray(volume.shape)[:, None, None] - 0.5
    return map_coordinates(volume, indices, order=1, mode="nearest")


def display_rgb(images, low, high, difference=False):
    from matplotlib import colormaps

    parts = []
    for image in images:
        level = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
        if difference:
            rgb = (colormaps["inferno"](level)[..., :3] * 255).astype("u1")
        else:
            rgb = np.repeat(np.rint(level * 255).astype("u1")[..., None], 3, axis=2)
        parts.append(rgb)
    return np.concatenate(parts, axis=1)


class AnalysisViewer:
    def __init__(self, args, parameters, source, camera_config):
        import json

        from analysis_backend import AnalysisBackend

        self.args, self.source = args, source
        self.model = EditableGaussians(parameters)
        self.query = None
        self.candidates = (np.array([], dtype=int), np.array([]), np.array([]), 0.0)
        self.roi_mode = False
        self.roi_anchor = None
        self.main_mode = "gaussians"
        self.slice_source = 0
        self.comparison_source = 0
        self.show_planes = True
        self.aligned_axis = 0
        self.show_help = False
        self.gaussian_style = getattr(args, "gaussian_style", "solid")
        self.mode_buttons = {}
        self.main_sliders = {}
        self.surface_cache = None
        self.surface_threshold = getattr(args, "iso_value", 0.05)
        self.direct_plane = None
        self.direct_actor = None
        root = Path(__file__).resolve().parents[1]
        metadata_path = args.metadata or root / "meta_data.json"
        meta = json.loads(metadata_path.read_text())
        self.bbox = np.asarray(meta["bbox"], dtype=float)
        self.shape = tuple(meta["scanner"]["nVoxel"])
        if len(self.shape) != 3 or any(int(n) != n or n < 2 for n in self.shape):
            raise ValueError("metadata scanner.nVoxel must contain three integers >= 2")
        self.positions = self.bbox[0] + (np.asarray(self.shape) // 2 + 0.5) / np.asarray(
            self.shape
        ) * (self.bbox[1] - self.bbox[0])
        self.resolution = args.analysis_resolution
        self.references = {}
        defaults = [
            (
                "FaCT-GS fixed",
                args.fact_volume or (root / "vol_pred.tiff" if args.method == "fact_gs" else None),
            ),
            (
                "GT fixed",
                args.gt_volume
                or (root / "vol_gt.npy" if (root / "vol_gt.npy").is_file() else None),
            ),
            ("R2 fixed", args.r2_volume),
        ]
        for label, path in defaults:
            if path is not None:
                if not Path(path).is_file() and label == "FaCT-GS fixed" and not args.fact_volume:
                    continue
                self.references[label] = load_reference(path, self.shape)
        self.geometry_label = next(iter(self.references), None)
        self.plotter = pv.Plotter(
            shape=(2, 3),
            window_size=camera_config.get("window_size", (1800, 1000)),
            title="CT Gaussian analysis",
            border=False,
        )
        for renderer, viewport in zip(
            self.plotter.renderers,
            (
                (0, 0.46, 0.58, 1),
                (0.58, 0.46, 0.8, 1),
                (0.8, 0.46, 1, 1),
                (0, 0, 1 / 3, 0.46),
                (1 / 3, 0, 2 / 3, 0.46),
                (2 / 3, 0, 1, 0.46),
            ),
        ):
            renderer.SetViewport(*viewport)
        self.plotter.set_background("#202533", all_renderers=True)
        self.plotter.theme.font.color = "white"
        self.plotter.subplot(0, 0)
        if self.gaussian_style == "solid":
            self.plotter.set_background("#252c38", top="#101318", all_renderers=False)
        self.plotter.ren_win.Render()
        # A separate process owns EGL; input callbacks never switch VTK's context.
        self.gpu = AnalysisBackend(parameters, self.bbox)
        self.minimum_density, self.maximum_ratio = args.min_density, args.max_ratio
        self.show_all = False
        self.slice_baseline, self.slice_current, self.reference_images = {}, {}, {}
        self.sliders = []
        self.camera_destination = args.camera_save or args.output / "camera.json"
        self.images = {}
        self.image_actors = {}
        self.camera_stamp = None
        self.busy = False
        from concurrent.futures import ThreadPoolExecutor

        self.edit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gaussian-edit")
        self.edit_future = None
        self.edit_timer = None
        self.closing = False
        self.refresh_slices()
        self.gray_range = list(args.gray_range or (0.0, 1.0))
        self.default_gray_range = self.gray_range.copy()
        self.gray_sliders = []
        self.draw_gaussians()
        self.plotter.subplot(0, 0)
        self.plotter.add_axes()
        self.draw_planes()
        self.plotter.reset_camera()
        from camera_config import aligned_camera

        config = {
            **aligned_camera(0, False, self.bbox, self.positions, self.plotter.renderer.GetSize()),
            **camera_config,
            "projection": "orthographic",
        }
        apply_pyvista_camera(self.plotter, config)
        self.draw_slices()
        self.refresh_aligned_views()
        self.draw_inspector()
        self.add_controls()
        self.plotter.subplot(0, 0)

    def label(self, text, name, x=0.02, y=0.95, size=10):
        name = f"panel_{self.plotter.renderers.active_index}_{name}"
        actor = self.plotter.add_text(
            text.replace("|", "/"),
            name=name,
            position=(x, y),
            viewport=True,
            font_size=size,
            color="white",
            render=False,
        )
        actor.GetTextProperty().SetVerticalJustificationToTop()

    def draw_image(self, panel, rgb, labels, aspect=1.0):
        self.plotter.subplot(*panel)
        h, w = rgb.shape[:2]
        first = panel not in self.images
        if first:
            grid = pv.ImageData(dimensions=(w + 1, h + 1, 1), spacing=(aspect / h, 1 / h, 1))
            grid.cell_data["rgb"] = rgb.reshape(-1, 3)
            self.images[panel] = grid
            self.image_actors[panel] = self.plotter.add_mesh(
                grid,
                scalars="rgb",
                rgb=True,
                lighting=False,
                show_scalar_bar=False,
                name="image",
                pickable=True,
            )
            self.plotter.view_xy()
            self.plotter.enable_parallel_projection()
            self.plotter.reset_camera()
            width, height = self.plotter.renderer.GetSize()
            panel_aspect = width / max(height, 1)
            self.plotter.camera.parallel_scale = (
                max(1 / 0.74, (w * aspect / h) / (0.94 * panel_aspect)) / 2
            )
        else:
            self.images[panel].cell_data["rgb"] = rgb.reshape(-1, 3)
        for i, label in enumerate(labels):
            self.label(label, f"label_{i}", x=(i + 0.05) / len(labels), y=0.94, size=8)

    def refresh_slices(self, axes=(0, 1, 2)):
        with self.gpu:
            self.gpu.update_density(self.model.parameters["density"])
            for axis in axes:
                self.slice_baseline[axis] = self.gpu.slice(
                    axis, self.positions[axis], self.resolution
                )
            self.gpu.update_density(self.model.density)
            for axis in axes:
                self.slice_current[axis] = self.gpu.slice(
                    axis, self.positions[axis], self.resolution
                )
                self.reference_images[axis] = {
                    label: reference_slice(
                        v, axis, self.positions[axis], self.bbox, self.resolution
                    )
                    for label, v in self.references.items()
                }

    def draw_slices(self):
        diffs = {i: np.maximum(self.slice_baseline[i] - self.slice_current[i], 0) for i in range(3)}
        delta_max = max(float(d.max()) for d in diffs.values())
        for axis in range(3):
            options = [
                ("GS edited", self.slice_current[axis]),
                *self.reference_images[axis].items(),
                ("Removed density", diffs[axis]),
            ]
            name, image = options[self.slice_source % len(options)]
            difference = name == "Removed density"
            rgb = display_rgb(
                [image],
                0 if difference else self.gray_range[0],
                delta_max if difference else self.gray_range[1],
                difference=difference,
            )
            index = int(
                round(
                    (self.positions[axis] - self.bbox[0, axis])
                    / (self.bbox[1, axis] - self.bbox[0, axis])
                    * self.shape[axis]
                    - 0.5
                )
            )
            names = [f"{'XYZ'[axis]} slice ONLY / index {index} / {name}"]
            remaining = [i for i in range(3) if i != axis]
            extent = self.bbox[1] - self.bbox[0]
            self.draw_image((1, axis), rgb, names, extent[remaining[0]] / extent[remaining[1]])
            self.label(
                f"{'XYZ'[remaining[0]]} right / {'XYZ'[remaining[1]]} up | "
                f"gray [{self.gray_range[0]:.3g}, {self.gray_range[1]:.3g}]",
                "axis_info",
                x=0.02,
                y=0.88,
                size=8,
            )

    def draw_planes(self):
        self.plotter.subplot(0, 0)
        for axis, color in enumerate(("red", "green", "cyan")):
            other = [i for i in range(3) if i != axis]
            points = np.zeros((4, 3))
            points[:, axis] = self.positions[axis]
            points[:, other[0]] = self.bbox[[0, 1, 1, 0], other[0]]
            points[:, other[1]] = self.bbox[[0, 0, 1, 1], other[1]]
            if self.main_mode == "render":
                direction = np.array(self.plotter.camera.position) - self.plotter.camera.focal_point
                points += direction / max(np.linalg.norm(direction), 1e-12) * 1e-4
            actor = self.plotter.add_mesh(
                pv.lines_from_points(points, close=True),
                color=color,
                line_width=2,
                name=f"plane_{axis}",
                pickable=False,
                reset_camera=False,
                render=False,
            )
            actor.visibility = self.show_planes and axis == self.aligned_axis

    def draw_gaussians(self):
        self.plotter.subplot(0, 0)
        if self.main_mode != "gaussians":
            for name in ("gaussians", "selected", "candidates"):
                self.plotter.remove_actor(name, render=False)
            if self.main_mode == "render":
                self.plotter.remove_actor("main_surface", render=False)
                self.draw_direct_view()
            else:
                self.remove_direct_view()
                self.draw_tiff_view()
            return
        self.remove_direct_view()
        self.plotter.remove_actor("main_surface", render=False)
        self.draw_gaussian_mesh()

    def draw_gaussian_mesh(self):
        from density_viewer import ellipsoid_mesh

        self.plotter.subplot(0, 0)
        p = self.model.parameters
        ids, matched = slice_gaussian_ids(
            p,
            self.aligned_axis,
            self.positions[self.aligned_axis],
            self.model.hidden,
            self.minimum_density,
            self.maximum_ratio,
            0 if self.show_all else self.args.max_count,
        )
        mesh, indices, _ = ellipsoid_mesh({k: v[ids] for k, v in p.items()}, 0, 1e12, 0)
        self.plotter.remove_actor("gaussians", render=False)
        if mesh.n_points:
            mesh["gaussian_id"] = ids[mesh["gaussian_id"]]
            if self.gaussian_style == "solid":
                self.plotter.add_mesh(
                    mesh,
                    color="#d4d4d4",
                    opacity=1.0,
                    ambient=0.2,
                    diffuse=0.8,
                    specular=0.15,
                    specular_power=20,
                    smooth_shading=False,
                    show_scalar_bar=False,
                    name="gaussians",
                    reset_camera=False,
                    render=False,
                )
            else:
                self.plotter.add_mesh(
                    mesh,
                    scalars="density",
                    cmap="inferno",
                    clim=(0, max(float(p["density"].max()), 1e-6)),
                    opacity=0.55,
                    show_scalar_bar=False,
                    name="gaussians",
                    reset_camera=False,
                    render=False,
                )
        self.label(
            f"{'XYZ'[self.aligned_axis]} slice Gaussians {len(indices)}/{matched} / x,y,z: align / Shift: reverse",
            "structure_title",
            size=9,
        )
        self.plotter.remove_actor("selected", render=False)
        i = self.model.selected
        self.plotter.remove_actor("candidates", render=False)
        candidate_ids = self.candidates[0]
        if len(candidate_ids):
            candidates, _, _ = ellipsoid_mesh(
                {k: v[candidate_ids] for k, v in p.items()}, 0, 1e12, 0
            )
            candidates["gaussian_id"] = candidate_ids[candidates["gaussian_id"]]
            self.plotter.add_mesh(
                candidates,
                color="#ffdf00",
                style="wireframe",
                line_width=2,
                name="candidates",
                pickable=True,
                reset_camera=False,
                render=False,
            )
        if i is not None:
            selected, _, _ = ellipsoid_mesh({k: v[i : i + 1] for k, v in p.items()}, 0, 1e12, 0)
            selected["gaussian_id"][:] = i
            self.plotter.add_mesh(
                selected,
                color="cyan",
                style="wireframe",
                line_width=3,
                name="selected",
                pickable=not self.model.hidden[i],
                reset_camera=False,
                render=False,
            )

    def remove_direct_view(self):
        self.plotter.remove_actor("main_direct", render=False)
        self.direct_actor = self.direct_plane = None

    def draw_direct_view(self, camera=None, density=None):
        from gs_renderer import camera_basis, configure_camera

        self.plotter.subplot(0, 0)
        camera = camera or capture_pyvista_camera(self.plotter)
        camera["projection"] = "orthographic"
        width, height = self.plotter.renderer.GetSize()
        if min(width, height) <= 0:
            return
        factor = min(1.0, 512 / max(width, height))
        size = (max(16, round(width * factor)), max(16, round(height * factor)))
        if density is None:
            with self.gpu:
                self.gpu.update_density(self.model.density)
                density = self.gpu.projection(camera, size)
        rgb = display_rgb([1 - np.exp(-self.args.exposure * density)], 0, 1)
        target = np.asarray(camera["target"], dtype=float)
        state = dict(yaw=0.0, pitch=0.0, roll=0.0, distance=3.0, span=1.0)
        configure_camera(camera, state, target)
        right, up = camera_basis(state["yaw"], state["pitch"], state["roll"])
        right = right * state["span"] * width / height
        up = up * state["span"]
        points = np.array(
            [target - right - up, target + right - up, target + right + up, target - right + up]
        )
        texture = pv.numpy_to_texture(rgb[::-1].copy())
        if self.direct_plane is None:
            self.direct_plane = pv.PolyData(points, [4, 0, 1, 2, 3])
            self.direct_plane.active_texture_coordinates = np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            )
            self.direct_actor = self.plotter.add_mesh(
                self.direct_plane,
                texture=texture,
                lighting=False,
                name="main_direct",
                pickable=False,
                reset_camera=False,
                render=False,
            )
        else:
            self.direct_plane.points = points
            self.direct_actor.texture = texture
        self.label(
            "GS direct rendering / X,Y,Z: align; Shift: reverse\n"
            "Drag to orbit / density projection / editable Gaussians",
            "structure_title",
            size=9,
        )
        self.draw_planes()

    def draw_tiff_view(self):
        self.plotter.subplot(0, 0)
        self.plotter.remove_actor("main_surface", render=False)
        if self.geometry_label is None:
            self.label(
                "No fixed volume loaded. Use --fact-volume or --r2-volume.",
                "structure_title",
                size=9,
            )
            return
        if self.surface_cache is None:
            volume = self.references[self.geometry_label]
            spacing = (self.bbox[1] - self.bbox[0]) / np.array(volume.shape)
            grid = pv.ImageData(
                dimensions=volume.shape, spacing=spacing, origin=self.bbox[0] + spacing / 2
            )
            grid["density"] = volume.ravel(order="F")
            self.surface_cache = grid.contour([self.surface_threshold], scalars="density")
        if self.surface_cache.n_points:
            self.plotter.add_mesh(
                self.surface_cache,
                color="#e2d6c3",
                smooth_shading=True,
                name="main_surface",
                pickable=False,
                reset_camera=False,
                render=False,
            )
        self.label(
            f"TIFF geometry ({self.geometry_label}) / X,Y,Z: align; Shift: reverse\n"
            f"Fixed reference / isovalue {self.surface_threshold:.4g}; Gaussian hides do not edit TIFF",
            "structure_title",
            size=8,
        )
        if not self.surface_cache.n_points:
            self.label(
                "No surface at this isovalue; adjust the TIFF isovalue slider.",
                "surface_status",
                x=0.02,
                y=0.73,
                size=8,
            )
        else:
            self.plotter.remove_actor("panel_0_surface_status", render=False)

    def set_main_mode(self, mode):
        if self.busy:
            return
        if mode not in ("gaussians", "render", "tiff"):
            raise ValueError("Unknown main view mode")
        self.main_mode = mode
        self.plotter.subplot(0, 0)
        if mode != "tiff":
            self.plotter.remove_actor("panel_0_surface_status", render=False)
        if mode == "gaussians" and self.gaussian_style == "solid":
            self.plotter.set_background("#252c38", top="#101318", all_renderers=False)
        else:
            self.plotter.set_background("#202533", all_renderers=False)
        for name, button in self.mode_buttons.items():
            button.GetRepresentation().SetState(name == mode)
        for name, widget in self.main_sliders.items():
            widget.SetDefaultRenderer(self.plotter.renderers[0])
            widget.SetCurrentRenderer(self.plotter.renderers[0])
            widget.SetEnabled(name == mode)
        self.draw_gaussians()
        self.draw_planes()
        actor = self.plotter.renderer.actors.get("probe_world")
        if actor is not None:
            actor.visibility = mode != "render"
        self.plotter.render()

    def align_main_camera(self, axis, reverse=False):
        from camera_config import aligned_camera

        if self.busy:
            return
        self.busy = True
        try:
            if axis != self.aligned_axis:
                self.query = None
                self.candidates = (np.array([], dtype=int), np.array([]), np.array([]), 0.0)
                self.model.selected = None
                self.roi_anchor = None
                self.clear_probe()
            self.plotter.subplot(0, 0)
            config = aligned_camera(
                axis, reverse, self.bbox, self.positions, self.plotter.renderer.GetSize()
            )
            apply_pyvista_camera(self.plotter, config)
            self.aligned_axis = axis
            if self.main_mode == "gaussians":
                self.draw_gaussians()
            self.draw_planes()
            self.refresh_aligned_views()
            self.draw_inspector()
            self.plotter.render()
        finally:
            self.busy = False

    def main_interaction(self, *unused):
        if self.main_mode != "render" or self.busy:
            return
        self.busy = True
        try:
            self.draw_direct_view()
            self.plotter.render()
        finally:
            self.busy = False

    def refresh_aligned_views(self):
        self.plotter.subplot(0, 0)
        camera = capture_pyvista_camera(self.plotter)
        camera["projection"] = "orthographic"
        self.camera_stamp = tuple(
            camera["position"] + camera["target"] + camera["up"] + [camera["parallel_scale"]]
        )
        self.draw_slice_comparison()
        if self.main_mode == "render":
            self.draw_direct_view(camera)

    def draw_slice_comparison(self):
        axis = self.aligned_axis
        before, after = self.slice_baseline[axis], self.slice_current[axis]
        delta = np.maximum(before - after, 0)
        max_delta = float(delta.max())
        index = self.comparison_source % 3
        image = (after, before, delta)[index]
        rgb = display_rgb(
            [image],
            0 if index == 2 else self.gray_range[0],
            max_delta if index == 2 else self.gray_range[1],
            difference=index == 2,
        )
        voxel_index = int(
            round(
                (self.positions[axis] - self.bbox[0, axis])
                / (self.bbox[1, axis] - self.bbox[0, axis])
                * self.shape[axis]
                - 0.5
            )
        )
        extent = self.bbox[1] - self.bbox[0]
        other = [i for i in range(3) if i != axis]
        self.draw_image(
            (0, 1),
            rgb,
            [f"{'XYZ'[axis]} slice {voxel_index} / " + ("after", "before", "difference")[index]],
            extent[other[0]] / extent[other[1]],
        )
        self.label(
            "T: after / before / difference\nSame layer as the bottom slice",
            "slice_comparison_info",
            x=0.02,
            y=0.88,
            size=7,
        )

    def draw_inspector(self):
        self.plotter.subplot(0, 2)
        selected = self.model.selected_parameters()
        if selected is None:
            details = "Right-click a slice pixel or Gaussian."
        else:
            details = (
                f"ID: {selected['original_id']} | "
                f"{'HIDDEN' if selected['hidden'] else 'active'}\n"
                f"Density: {selected['density']:.7g} | ratio: {selected['aspect_ratio']:.5g}\n"
                f"Center: " + ", ".join(f"{v:.3g}" for v in selected["center"]) + "\n"
                "Scales: " + ", ".join(f"{v:.3g}" for v in selected["scales"])
            )
        hidden = self.model.original_ids[self.model.hidden]
        query_text = "\nSlice probe: none"
        if self.query:
            query_text = f"\n{self.query['kind']} probe | {'XYZ'[self.query['axis']]}={self.query['coordinate']:.5g}"
            if self.query["kind"] == "point":
                query_text += "\nXYZ: " + ", ".join(f"{x:.5g}" for x in self.query["point"])
            else:
                query_text += f" | raster ROI {self.query['rectangle']}"
            ids, values, fractions, total = self.candidates
            query_text += f"\nActive density contribution total: {total:.6g}\nRank / Original ID / contribution / share of all"
            query_text += "".join(
                f"\n{rank + 1:>2}  {self.model.original_ids[i]:>7}  {value:.5g}  {fraction * 100:.2f}%"
                for rank, (i, value, fraction) in enumerate(zip(ids[:5], values[:5], fractions[:5]))
            )
            if len(ids) > 5:
                query_text += f"\nShowing 5/{len(ids)} candidates; N/P cycles all."
            if not len(ids):
                query_text += "\nNo active contributing Gaussians at this probe."
        text = (
            details
            + "\n"
            + f"Hidden: {len(hidden)} | IDs: {hidden[:8].tolist()}"
            + (" ..." if len(hidden) > 8 else "")
            + "\n"
            + query_text
            + "\n"
            + f"B: region picking {'ON (right-click two corners)' if self.roi_mode else 'off'}\n"
            "N/P: next/previous contribution candidate (yellow)\n"
            "H: hide selected / G: hide candidate group / U: undo\n"
            "R: restore all | J: move slices to selection\n"
            "A: capped/all ellipsoids | S: save analysis\n"
            "1/2/3: Gaussian / GS render / TIFF (top-left)\n"
            "X/Y/Z: align with slice / Shift: reverse view\n"
            "D: gray solid / density colors (Gaussian mode)\n"
            "C: save camera | L: load camera\n"
            "Gray range shared across GS and fixed references.\n"
            "XYZ model axes; pre-aligned references; GS support: 3 sigma.\n"
        )
        if not self.show_help:
            text = (
                details + f"\nHidden: {len(hidden)}\n" + query_text + "\n\n"
                "1/2/3: main view mode\n"
                "x/y/z: align; Shift: reverse\n"
                "V: slice reference / difference\n"
                "T: slice after / before / difference\n"
                "H: hide / U: undo / R: restore\n"
                "W: reset gray range\n"
                "F1: all controls"
            )
        if self.edit_future is not None:
            text = "Updating after edit...\nWindow remains interactive.\n\n" + text
        self.label(text, "inspector", x=0.025, y=0.98, size=7)

    def edit_changed(self, operation):
        if self.busy:
            return
        if not operation():
            return
        self.busy = True
        self.plotter.subplot(0, 0)
        camera = capture_pyvista_camera(self.plotter)
        width, height = self.plotter.renderer.GetSize()
        factor = min(1.0, 512 / max(width, height, 1))
        direct_size = (
            (max(16, round(width * factor)), max(16, round(height * factor)))
            if self.main_mode == "render"
            else None
        )
        hidden = self.model.hidden.copy()
        positions = self.positions.copy()
        query = dict(self.query) if self.query else None
        self.edit_future = self.edit_executor.submit(
            self.compute_edit, hidden, positions, query, camera, direct_size
        )
        self.draw_inspector()
        self.plotter.render()
        self.edit_timer = self.plotter.iren.interactor.CreateRepeatingTimer(30)

    def compute_edit(self, hidden, positions, query, camera, direct_size):
        density = np.where(hidden, 0, self.model.parameters["density"])
        with self.gpu:
            self.gpu.update_density(density)
            slices = {
                axis: self.gpu.slice(axis, positions[axis], self.resolution) for axis in range(3)
            }
            direct = self.gpu.projection(camera, direct_size) if direct_size else None
        candidates = None
        if query:
            if query["kind"] == "point":
                scores = point_contributions(self.model.parameters, query["point"], hidden)
            else:
                scores = region_contributions(
                    self.model.parameters,
                    query["axis"],
                    query["coordinate"],
                    self.bbox,
                    self.resolution,
                    query["rectangle"],
                    hidden,
                )
            candidates = rank_contributions(scores, self.args.candidate_count)
        return slices, direct, candidates, camera

    def finish_edit(self, *unused):
        if self.closing or self.edit_future is None or not self.edit_future.done():
            return
        future, self.edit_future = self.edit_future, None
        if self.edit_timer is not None:
            self.plotter.iren.interactor.DestroyTimer(self.edit_timer)
            self.edit_timer = None
        try:
            slices, direct, candidates, camera = future.result()
            self.slice_current = slices
            if candidates is not None:
                self.candidates = candidates
            if self.main_mode == "render":
                self.draw_direct_view(camera, density=direct)
            else:
                self.draw_gaussians()
            self.draw_slices()
            self.draw_slice_comparison()
        except Exception as error:
            print(f"Edit update failed: {error}")
            self.plotter.subplot(0, 2)
            self.label(f"Edit update failed:\n{error}", "edit_error", x=0.02, y=0.7, size=8)
        finally:
            self.busy = False
            self.draw_inspector()
            self.plotter.render()

    def pick(self, point, picker):
        if self.busy:
            return
        dataset = picker.GetDataSet()
        if dataset is None:
            return
        mesh = pv.wrap(dataset)
        cell_id = picker.GetCellId()
        panel = next(
            (panel for panel, actor in self.image_actors.items() if actor == picker.GetActor()),
            None,
        )
        if panel is not None:
            if panel[0] == 1:
                self.pick_slice(panel[1], picker.GetPickPosition())
            elif panel == (0, 1):
                self.pick_slice(self.aligned_axis, picker.GetPickPosition())
            return
        if "gaussian_id" not in mesh.point_data or cell_id < 0:
            return
        point_id = mesh.get_cell(cell_id).point_ids[0]
        self.model.select(mesh["gaussian_id"][point_id])
        self.draw_gaussians()
        self.draw_inspector()
        self.plotter.render()

    def pick_slice(self, axis, image_point):
        if self.busy:
            return
        if axis != self.aligned_axis:
            self.align_main_camera(axis)
        remaining = [i for i in range(3) if i != axis]
        extent = self.bbox[1] - self.bbox[0]
        aspect = extent[remaining[0]] / extent[remaining[1]]
        x, y = image_point[:2]
        if y < 0 or y >= 1 or x < 0:
            return
        tile = int(x / aspect)
        column = int(np.clip((x / aspect - tile) * self.resolution, 0, self.resolution - 1))
        row = int(np.clip(y * self.resolution, 0, self.resolution - 1))
        if self.roi_mode:
            if self.roi_anchor is None or self.roi_anchor[0] != axis:
                self.roi_anchor = (axis, row, column, tile)
                print("ROI first corner set; right-click the second corner on the same slice.")
                self.draw_probe(axis, (row, row, column, column))
                self.plotter.render()
                return
            _, r0, c0, _ = self.roi_anchor
            rectangle = (min(r0, row), max(r0, row), min(c0, column), max(c0, column))
            self.query = dict(
                kind="region",
                axis=axis,
                coordinate=float(self.positions[axis]),
                rectangle=rectangle,
            )
            self.roi_anchor = None
            self.draw_probe(axis, rectangle)
        else:
            world = pixel_world(axis, row, column, self.positions[axis], self.bbox, self.resolution)
            self.query = dict(
                kind="point",
                axis=axis,
                coordinate=float(self.positions[axis]),
                point=world.tolist(),
                pixel=(row, column),
            )
            self.draw_probe(axis, (row, row, column, column))
        self.update_candidates(select_first=True)
        self.draw_gaussians()
        self.draw_inspector()
        self.plotter.render()

    def draw_probe(self, axis, rectangle):
        r0, r1, c0, c1 = rectangle
        self.plotter.subplot(1, axis)
        remaining = [i for i in range(3) if i != axis]
        extent = self.bbox[1] - self.bbox[0]
        aspect = extent[remaining[0]] / extent[remaining[1]]
        for tile in range(1):
            x0, x1 = (
                aspect * (tile + c0 / self.resolution),
                aspect * (tile + (c1 + 1) / self.resolution),
            )
            y0, y1 = r0 / self.resolution, (r1 + 1) / self.resolution
            points = np.array([[x0, y0, 0.01], [x1, y0, 0.01], [x1, y1, 0.01], [x0, y1, 0.01]])
            self.plotter.add_mesh(
                pv.lines_from_points(points, close=True),
                color="cyan",
                line_width=2,
                name=f"probe_{tile}",
                pickable=False,
                reset_camera=False,
                render=False,
            )
        point = pixel_world(
            axis, (r0 + r1) / 2, (c0 + c1) / 2, self.positions[axis], self.bbox, self.resolution
        )
        self.plotter.subplot(0, 0)
        actor = self.plotter.add_mesh(
            pv.Sphere(center=point, radius=float(extent.max()) * 0.008),
            color="cyan",
            name="probe_world",
            pickable=False,
            reset_camera=False,
            render=False,
        )
        actor.visibility = self.main_mode != "render"

    def update_candidates(self, select_first=False):
        if self.query is None:
            return
        q = self.query
        if q["kind"] == "point":
            scores = point_contributions(self.model.parameters, q["point"], self.model.hidden)
        else:
            scores = region_contributions(
                self.model.parameters,
                q["axis"],
                q["coordinate"],
                self.bbox,
                self.resolution,
                q["rectangle"],
                self.model.hidden,
            )
        self.candidates = rank_contributions(scores, self.args.candidate_count)
        if select_first and len(self.candidates[0]):
            self.model.select(self.candidates[0][0])

    def cycle_candidate(self, step):
        if self.busy:
            return
        ids = self.candidates[0]
        if not len(ids):
            return
        current = np.flatnonzero(ids == self.model.selected)
        index = (int(current[0]) + step) % len(ids) if len(current) else 0
        self.model.select(ids[index])
        self.draw_gaussians()
        self.draw_inspector()
        self.plotter.render()

    def slice_changed(self, axis, value):
        if self.busy or abs(self.positions[axis] - value) < 1e-9:
            return
        self.positions[axis] = value
        self.model.selected = None
        self.query = None
        self.candidates = (np.array([], dtype=int), np.array([]), np.array([]), 0.0)
        self.roi_anchor = None
        self.clear_probe()
        self.refresh_slices((axis,))
        self.draw_slices()
        self.draw_planes()
        self.align_main_camera(axis)
        self.draw_inspector()
        self.plotter.render()

    def jump_to_selected(self):
        if self.busy:
            return
        if self.model.selected is None:
            return
        center = self.model.parameters["centers"][self.model.selected]
        self.query = None
        self.candidates = (np.array([], dtype=int), np.array([]), np.array([]), 0.0)
        self.roi_anchor = None
        self.clear_probe()
        for axis in range(3):
            self.positions[axis] = np.clip(center[axis], *self.slice_ranges[axis])
            index = (self.positions[axis] - self.bbox[0, axis]) / (
                self.bbox[1, axis] - self.bbox[0, axis]
            ) * self.shape[axis] - 0.5
            self.sliders[axis].GetRepresentation().SetValue(index)
        self.refresh_slices()
        self.draw_slices()
        self.draw_slice_comparison()
        self.draw_planes()
        self.draw_gaussians()
        self.draw_inspector()
        self.plotter.render()

    def clear_probe(self):
        for axis in range(3):
            self.plotter.subplot(1, axis)
            for tile in range(2 + len(self.references)):
                self.plotter.remove_actor(f"probe_{tile}", render=False)
        self.plotter.subplot(0, 0)
        self.plotter.remove_actor("probe_world", render=False)

    def interaction_ended(self, *unused):
        if self.busy:
            return
        self.plotter.subplot(0, 0)
        c = capture_pyvista_camera(self.plotter)
        stamp = tuple(c["position"] + c["target"] + c["up"] + [c["parallel_scale"]])
        if stamp != self.camera_stamp:
            self.busy = True
            try:
                self.refresh_aligned_views()
                self.plotter.render()
            finally:
                self.busy = False

    def save(self):
        if self.busy:
            return
        import json

        destination = self.args.output
        self.model.export(destination, self.source)
        self.plotter.subplot(0, 0)
        save_camera(self.camera_destination, capture_pyvista_camera(self.plotter))
        self.plotter.screenshot(destination / "analysis.png")
        np.savez_compressed(
            destination / "analysis_slices.npz",
            **{f"edited_{'xyz'[i]}": self.slice_current[i] for i in range(3)},
            **{f"baseline_{'xyz'[i]}": self.slice_baseline[i] for i in range(3)},
        )
        (destination / "analysis_state.json").write_text(
            json.dumps(
                dict(
                    positions=self.positions.tolist(),
                    bbox=self.bbox.tolist(),
                    resolution=self.resolution,
                    gray_range=self.gray_range,
                    truncation_sigma=3,
                    main_mode=self.main_mode,
                    gaussian_style=self.gaussian_style,
                    tiff_isovalue=self.surface_threshold,
                    references=list(self.references),
                    mesh_min_density=self.minimum_density,
                    mesh_max_ratio=self.maximum_ratio,
                ),
                indent=2,
            )
        )
        if self.query is not None:
            ids, values, fractions, total = self.candidates
            (destination / "contributions.json").write_text(
                json.dumps(
                    dict(
                        query=self.query,
                        support_sigma=3,
                        total=total,
                        original_ids=self.model.original_ids[ids].tolist(),
                        source_rows=ids.tolist(),
                        contributions=values.tolist(),
                        fractions_of_all=fractions.tolist(),
                    ),
                    indent=2,
                )
            )
        print(f"Saved analysis to {destination}")

    def add_controls(self):
        # Own the keyboard completely. PyVista appends callbacks (e.g. v/b/p),
        # and VTK handles CharEvent separately (3=stereo, w=wireframe, r=reset).
        self.plotter.iren.clear_key_event_callbacks()
        interactor = self.plotter.iren.interactor

        def suppress_native_char(obj, event):
            obj.GetCommand(self._char_observer).SetAbortFlag(1)

        self._char_observer = interactor.AddObserver("CharEvent", suppress_native_char, 1.0)
        self.plotter.subplot(0, 0)
        self.plotter.enable_surface_point_picking(
            callback=self.pick,
            use_picker=True,
            picker="cell",
            show_point=False,
            show_message=False,
            left_clicking=False,
        )
        self.plotter.iren.clear_key_event_callbacks()
        self.plotter.add_key_event("q", self.plotter.iren.terminate_app)
        for key, callback in dict(
            h=lambda: self.edit_changed(self.model.hide_selected),
            g=lambda: self.edit_changed(lambda: self.model.hide_ids(self.candidates[0])),
            u=lambda: self.edit_changed(self.model.undo),
            r=lambda: self.edit_changed(self.model.restore_all),
            j=self.jump_to_selected,
            s=self.save,
        ).items():
            self.plotter.add_key_event(key, callback)

        def toggle_all():
            if self.busy:
                return
            self.show_all = not self.show_all
            self.draw_gaussians()
            self.plotter.render()

        def save_view():
            self.plotter.subplot(0, 0)
            save_camera(self.camera_destination, capture_pyvista_camera(self.plotter))

        def load_view():
            if self.busy:
                return
            try:
                config = load_camera(self.args.camera or self.camera_destination)
                if config.get("projection", "orthographic") != "orthographic":
                    raise ValueError("Analysis uses an orthographic camera")
                self.plotter.subplot(0, 0)
                apply_pyvista_camera(self.plotter, config)
                self.refresh_aligned_views()
                self.plotter.render()
            except (OSError, ValueError) as error:
                print(f"Camera load failed: {error}")

        self.plotter.add_key_event("a", toggle_all)
        self.plotter.add_key_event("c", save_view)
        self.plotter.add_key_event("l", load_view)
        self.plotter.add_key_event("n", lambda: self.cycle_candidate(1))
        self.plotter.add_key_event("p", lambda: self.cycle_candidate(-1))

        def toggle_region():
            self.roi_mode = not self.roi_mode
            self.roi_anchor = None
            self.draw_inspector()
            self.plotter.render()

        self.plotter.add_key_event("b", toggle_region)

        def cycle_slices():
            self.slice_source = (self.slice_source + 1) % (2 + len(self.references))
            self.draw_slices()
            self.plotter.render()

        def cycle_slice_comparison():
            if self.busy:
                return
            self.comparison_source = (self.comparison_source + 1) % 3
            self.draw_slice_comparison()
            self.plotter.render()

        def reset_gray():
            self.gray_range = self.default_gray_range.copy()
            for widget, value in zip(self.gray_sliders, self.gray_range):
                widget.GetRepresentation().SetValue(value)
            self.draw_slices()
            self.draw_slice_comparison()
            self.plotter.render()

        def toggle_help():
            self.show_help = not self.show_help
            self.draw_inspector()
            self.plotter.render()

        def toggle_planes():
            self.show_planes = not self.show_planes
            self.draw_planes()
            self.plotter.render()

        self.plotter.add_key_event("v", cycle_slices)
        self.plotter.add_key_event("t", cycle_slice_comparison)
        self.plotter.add_key_event("w", reset_gray)
        self.plotter.add_key_event("F1", toggle_help)
        self.plotter.add_key_event("f", toggle_planes)
        for key, mode in (("1", "gaussians"), ("2", "render"), ("3", "tiff")):
            self.plotter.add_key_event(key, lambda m=mode: self.set_main_mode(m))
        for axis, key in enumerate("xyz"):
            self.plotter.add_key_event(
                key, lambda a=axis: self.align_main_camera(a, bool(interactor.GetShiftKey()))
            )
            self.plotter.add_key_event(key.upper(), lambda a=axis: self.align_main_camera(a, True))

        def toggle_material():
            self.gaussian_style = "density" if self.gaussian_style == "solid" else "solid"
            self.set_main_mode(self.main_mode)

        self.plotter.add_key_event("d", toggle_material)
        self.plotter.subplot(0, 0)
        for i, (mode, title) in enumerate(
            (("gaussians", "1 Gaussians"), ("render", "2 GS render"), ("tiff", "3 TIFF geometry"))
        ):
            self.mode_buttons[mode] = self.plotter.add_radio_button_widget(
                lambda m=mode: self.set_main_mode(m),
                "main_view_mode",
                value=mode == self.main_mode,
                position=(10, 10),
                size=18,
                border_size=3,
                color_on="cyan",
                color_off="gray",
            )
            self.mode_buttons[mode].SetDefaultRenderer(self.plotter.renderers[0])
            self.label(title, f"mode_button_{mode}", x=i / 3 + 0.07, y=0.84, size=8)

        def place_buttons(*unused):
            renderer = self.plotter.renderers[0]
            width, height = renderer.GetSize()
            for i, button in enumerate(self.mode_buttons.values()):
                x, y = width * (i / 3 + 0.02), height * 0.81
                button.GetRepresentation().PlaceWidget((x, x + 18, y, y + 18, 0, 0))

        place_buttons()
        self.plotter.iren.add_observer("ConfigureEvent", place_buttons)
        self.slice_ranges = []
        for axis in range(3):
            self.plotter.subplot(1, axis)
            spacing = (self.bbox[1, axis] - self.bbox[0, axis]) / self.shape[axis]
            bounds = self.bbox[0, axis] + spacing / 2, self.bbox[1, axis] - spacing / 2
            self.slice_ranges.append(bounds)

            def index_changed(value, a=axis):
                index = int(np.clip(round(value), 0, self.shape[a] - 1))
                coordinate = self.bbox[0, a] + (index + 0.5) / self.shape[a] * (
                    self.bbox[1, a] - self.bbox[0, a]
                )
                self.slice_changed(a, coordinate)

            slider = self.plotter.add_slider_widget(
                index_changed,
                (0, self.shape[axis] - 1),
                value=self.shape[axis] // 2,
                title=f"{'XYZ'[axis]} slice index",
                color="white",
                fmt="%.0f",
                pointa=(0.12, 0.1),
                pointb=(0.88, 0.1),
                interaction_event="end",
                title_height=0.025,
            )
            self.sliders.append(slider)
        self.plotter.subplot(0, 0)

        def filter_changed(name, value):
            if self.busy:
                return
            setattr(self, name, value)
            self.draw_gaussians()
            self.plotter.render()

        maximum = max(float(self.model.parameters["density"].max()), 1e-6)
        self.main_sliders["gaussians"] = self.plotter.add_slider_widget(
            lambda v: filter_changed("minimum_density", v),
            (0, maximum),
            value=min(self.minimum_density, maximum),
            title="Slice peak density filter",
            color="white",
            pointa=(0.1, 0.08),
            pointb=(0.9, 0.08),
            interaction_event="end",
            title_height=0.025,
        )

        def surface_changed(value):
            if self.busy:
                return
            self.surface_threshold = value
            self.surface_cache = None
            if self.main_mode == "tiff":
                self.draw_tiff_view()
                self.plotter.render()

        self.plotter.subplot(0, 0)
        surface_max = max((float(v.max()) for v in self.references.values()), default=1.0)
        self.main_sliders["tiff"] = self.plotter.add_slider_widget(
            surface_changed,
            (0, max(surface_max, 1e-6)),
            value=min(self.surface_threshold, surface_max),
            title="TIFF isovalue",
            color="white",
            pointa=(0.1, 0.08),
            pointb=(0.9, 0.08),
            interaction_event="end",
            title_height=0.025,
        )

        def exposure_changed(value):
            if self.busy:
                return
            self.args.exposure = value
            if self.main_mode == "render":
                self.refresh_aligned_views()
                self.plotter.render()

        self.plotter.subplot(0, 0)
        self.main_sliders["render"] = self.plotter.add_slider_widget(
            exposure_changed,
            (0.05, max(20.0, self.args.exposure)),
            value=self.args.exposure,
            title="GS exposure",
            color="white",
            pointa=(0.1, 0.08),
            pointb=(0.9, 0.08),
            interaction_event="end",
            title_height=0.025,
        )
        for mode, widget in self.main_sliders.items():
            widget.SetDefaultRenderer(self.plotter.renderers[0])
            widget.SetCurrentRenderer(self.plotter.renderers[0])
            widget.SetEnabled(mode == self.main_mode)
        self.plotter.subplot(0, 2)

        def window_changed(index, value):
            other = self.gray_range[1 - index]
            if (index == 0 and value >= other) or (index == 1 and value <= other):
                return
            self.gray_range[index] = value
            self.draw_slices()
            self.draw_slice_comparison()
            self.plotter.render()

        limit = max(abs(self.gray_range[0]), abs(self.gray_range[1]), 1e-6) * 2
        for index, title, y in ((0, "Gray lower", 0.18), (1, "Gray upper", 0.07)):
            self.plotter.subplot(0, 2)
            widget = self.plotter.add_slider_widget(
                lambda v, i=index: window_changed(i, v),
                (-limit, limit),
                value=self.gray_range[index],
                title=title,
                color="white",
                pointa=(0.1, y),
                pointb=(0.9, y),
                interaction_event="end",
                title_height=0.025,
            )
            self.gray_sliders.append(widget)
        self.plotter.iren.add_observer("EndInteractionEvent", self.interaction_ended)
        self.plotter.iren.add_observer("InteractionEvent", self.main_interaction)
        self.plotter.iren.add_observer("TimerEvent", self.finish_edit)

    def show(self):
        try:
            print(
                "Analysis: right-click ellipsoid, H hide, U undo, R restore, J jump slices, S export."
            )
            print("Fixed reference volumes are assumed aligned in metadata XYZ coordinates.")
            self.plotter.show(auto_close=False)
        finally:
            self.closing = True
            self.edit_executor.shutdown(wait=False, cancel_futures=True)
            self.gpu.release()
            self.plotter.close()


def run(args, parameters, source, camera_config):
    AnalysisViewer(args, parameters, source, camera_config).show()
