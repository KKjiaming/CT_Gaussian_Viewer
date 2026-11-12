#!/usr/bin/env python3
"""CT 密度体渲染与高斯密度筛选器；不使用 RGB 3DGS 的颜色混合模型。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile

from camera_config import (add_camera_arguments, camera_from_args, apply_pyvista_camera,
                           capture_pyvista_camera, load_camera, save_camera)


def load_parameters(path):
    """读取已导出的、完成激活的高斯参数，不把 raw density 当成真实密度。"""
    with np.load(path, allow_pickle=False) as stored:
        parameters = {name: stored[name] for name in ("centers", "scales", "rotations", "density")}
        if 'original_ids' in stored:
            parameters['original_ids'] = stored['original_ids']
    count = len(parameters["centers"])
    expected = {"centers": (count, 3), "scales": (count, 3),
                "rotations": (count, 3, 3), "density": (count,)}
    for name, shape in expected.items():
        if parameters[name].shape != shape or not np.isfinite(parameters[name]).all():
            raise ValueError(f"Invalid {name}: expected finite array of shape {shape}")
    if count == 0 or np.any(parameters["scales"] <= 0) or np.any(parameters["density"] < 0):
        raise ValueError("Empty model, nonpositive scales or negative densities")
    if 'original_ids' in parameters:
        ids = parameters['original_ids']
        if ids.shape != (count,) or ids.dtype.kind not in 'iu' or np.any(ids<0) or len(np.unique(ids))!=count:
            raise ValueError('original_ids must contain one unique nonnegative integer per Gaussian')
    parameters.setdefault('original_ids',np.arange(count,dtype=np.int64))
    return parameters


def ellipsoid_mesh(parameters, minimum_density, maximum_ratio, maximum_count):
    """密度阈值按中心密度筛选；每个椭球显示其 1σ 外壳，不是组织表面。"""
    scales = parameters["scales"]
    density = parameters["density"]
    ratio = scales.max(axis=1) / scales.min(axis=1)
    indices = np.flatnonzero((density >= minimum_density) & (ratio <= maximum_ratio))
    matched = len(indices)
    if maximum_count > 0 and matched > maximum_count:
        indices = indices[np.argsort(density[indices])[-maximum_count:]]
    if len(indices) == 0:
        return pv.PolyData(), indices, matched
    sphere = pv.Icosphere(radius=1, nsub=1)
    vertices = np.einsum("nij,nvj->nvi", parameters["rotations"][indices],
                         sphere.points[None] * scales[indices, None])
    vertices += parameters["centers"][indices, None]
    base_faces = sphere.faces.reshape(-1, 4)[:, 1:]
    triangles = (base_faces[None] + np.arange(len(indices))[:, None, None] * sphere.n_points).reshape(-1, 3)
    faces = np.column_stack((np.full(len(triangles), 3), triangles))
    mesh = pv.PolyData(vertices.reshape(-1, 3), faces)
    mesh["density"] = np.repeat(density[indices], sphere.n_points)
    mesh["aspect_ratio"] = np.repeat(ratio[indices], sphere.n_points)
    mesh["gaussian_id"] = np.repeat(indices, sphere.n_points)
    mesh['original_id'] = np.repeat(parameters.get('original_ids',np.arange(len(density)))[indices],sphere.n_points)
    return mesh, indices, matched


def resolve_input(args, root):
    if args.input:
        return args.input.resolve()
    local = root / (f"{args.method}_parameters.npz" if args.mode in ("gaussians", "render", "analysis") else "vol_pred.tiff")
    if local.is_file():
        return local
    if args.mode in ("gaussians", "render", "analysis"):
        return root / f"output/compare/gaussians/meshes/{args.method}_parameters.npz"
    entries = json.loads((root / "output/compare/metrics.json").read_text())
    method = "FaCT-GS" if args.method == "fact_gs" else "R2-Gaussian"
    run = Path(next(entry["run"] for entry in entries if entry["method"] == method))
    prefix = "step_" if args.method == "fact_gs" else "iteration_"
    checkpoint = max((run / "point_cloud").glob(f"{prefix}*"), key=lambda path: int(path.name.split("_")[-1]))
    return checkpoint / ("vol_pred.tiff" if args.method == "fact_gs" else "vol_pred.npy")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gaussians", "render", "tiff", "analysis", "volume"], default="tiff",
                        help="gaussians: 高斯椭球；render: GPU 直接高斯投影；tiff: TIFF 几何；analysis: 切片贡献分析；volume: 兼容旧体渲染入口")
    parser.add_argument("--method", choices=["fact_gs", "r2_gaussian"], default="fact_gs")
    parser.add_argument("--input", type=Path, help="体积 NPY/TIFF 或激活后的高斯参数 NPZ")
    parser.add_argument("--output", type=Path, default=root / "output/compare/density_viewer")
    parser.add_argument("--min-density", type=float, default=0.05)
    parser.add_argument("--max-ratio", type=float, default=1000)
    parser.add_argument("--max-count", type=int, default=5000, help="椭球数量上限；0 为全部")
    parser.add_argument("--opacity", type=float, default=None)
    parser.add_argument('--gaussian-style',choices=['solid','density'],default='solid',
                        help='高斯球材质：solid 灰白实体；density 密度着色')
    parser.add_argument("--exposure", type=float, default=2.0, help="直接高斯渲染的曝光")
    parser.add_argument("--tiff-style", choices=["surface", "volume"], default="surface")
    parser.add_argument("--export-only", action="store_true", help="无窗口导出筛选后高斯网格和参数")
    parser.add_argument('--metadata',type=Path,help='分析模式的 bbox 与 nVoxel；默认本地 meta_data.json')
    parser.add_argument('--fact-volume',type=Path,help='已对齐的 FaCT-GS 固定参考体积')
    parser.add_argument('--gt-volume',type=Path,help='已对齐的 GT 固定参考体积')
    parser.add_argument('--r2-volume',type=Path,help='已对齐的 R2-Gaussian 固定参考体积')
    parser.add_argument('--analysis-resolution',type=int,default=256,help='动态切片及投影的像素分辨率；默认对齐原生 256 体素')
    parser.add_argument('--gray-range',type=float,nargs=2,metavar=('MIN','MAX'),help='所有密度切片共用的灰度范围')
    parser.add_argument('--candidate-count',type=int,default=8,help='切片拾取后高亮的贡献高斯数量')
    parser.add_argument('--iso-value',type=float,default=.05,help='分析窗口中 TIFF 几何的初始等值面阈值')
    add_camera_arguments(parser)
    args = parser.parse_args()
    if args.opacity is None:
        args.opacity = 1. if args.mode=='gaussians' and args.gaussian_style=='solid' else .2
    try:
        camera_config = camera_from_args(args)
        if args.mode in ('render','analysis') and camera_config.get('projection') == 'perspective':
            raise ValueError('直接高斯渲染与分析模式目前只支持 orthographic 正交相机')
        if not 16 <= args.analysis_resolution <= 2048:
            raise ValueError('--analysis-resolution must be between 16 and 2048')
        if not 1 <= args.candidate_count <= 100:
            raise ValueError('--candidate-count must be between 1 and 100')
        if not np.isfinite(args.iso_value) or args.iso_value<0:
            raise ValueError('--iso-value must be finite and nonnegative')
        if args.gray_range and (not np.isfinite(args.gray_range).all() or args.gray_range[1]<=args.gray_range[0]):
            raise ValueError('--gray-range requires finite MIN < MAX')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.min_density < 0 or args.max_ratio < 1 or args.max_count < 0 or not 0 <= args.opacity <= 1:
        parser.error("Invalid density, ratio, count or opacity")
    if args.export_only and args.mode != "gaussians":
        parser.error("--export-only requires --mode gaussians")
    if not np.isfinite(args.exposure) or args.exposure <= 0:
        parser.error("--exposure must be finite and positive")
    path = resolve_input(args, root)
    args.output.mkdir(parents=True, exist_ok=True)
    camera_destination = args.camera_save or args.output / 'camera.json'
    state = {"density": args.min_density, "ratio": args.max_ratio, "opacity": args.opacity}
    state["tiff_style"] = "volume" if args.mode == "volume" else args.tiff_style

    if args.mode == 'analysis':
        from analysis_viewer import run
        run(args,load_parameters(path),path,camera_config)
        return

    if args.mode == "render":
        from gs_renderer import run
        run(load_parameters(path), path, args.output, args.min_density, args.max_ratio, args.exposure,
            camera_config=camera_config, camera_destination=camera_destination,
            camera_source=args.camera)
        return

    if args.mode == "gaussians":
        parameters = load_parameters(path)
        density_max = float(parameters["density"].max())
        if args.export_only:
            mesh, indices, matched = ellipsoid_mesh(parameters, state["density"], state["ratio"], args.max_count)
            if not len(indices):
                raise ValueError("No Gaussians match the filters")
            mesh.save(args.output / f"{args.method}_filtered.vtp")
            np.savez_compressed(args.output / f"{args.method}_filtered.npz",
                                **{name: values[indices] for name, values in parameters.items()})
            print(f"Matched {matched}; exported {len(indices)} Gaussians to {args.output}")
            return
    else:
        volume = np.load(path, allow_pickle=False) if path.suffix == ".npy" else tifffile.imread(path)
        if volume.dtype == np.uint8:
            volume = volume.astype(np.float32) / 255
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError("Expected a finite 3D density volume")
        # 当前官方数据的归一化边界为 [-1,1]^3；体素中心与模型体素化一致。
        spacing = 2 / np.asarray(volume.shape)
        grid = pv.ImageData(dimensions=volume.shape, spacing=spacing, origin=-1 + spacing / 2)
        grid["density"] = volume.astype(np.float32).ravel(order="F")
        density_max = float(volume.max())

    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.set_background("#202533")
    if args.mode=='gaussians' and args.gaussian_style=='solid':
        plotter.set_background('#5555a2',top='#090914')
    plotter.theme.font.color = "white"
    plotter.add_axes()

    def redraw():
        plotter.remove_actor("density", render=False)
        if args.mode == "gaussians":
            mesh, indices, matched = ellipsoid_mesh(parameters, state["density"], state["ratio"], args.max_count)
            if len(indices):
                if args.gaussian_style=='solid':
                    plotter.add_mesh(mesh,name='density',color='#d4d4d4',opacity=state['opacity'],
                        ambient=.2,diffuse=.8,specular=.15,specular_power=20,smooth_shading=False,
                        reset_camera=False)
                else:
                    plotter.add_mesh(mesh, name="density", scalars="density", cmap="inferno",
                                     clim=(0, density_max), opacity=state["opacity"], reset_camera=False)
            message = (f"{path.name}\nMatched {matched}/{len(parameters['density'])}; "
                       f"displayed {len(indices)} 1-sigma ellipsoids\n"
                       "A: toggle all / capped count; S: export")
        elif state["tiff_style"] == "surface":
            surface = grid.contour([state["density"]], scalars="density")
            if surface.n_points:
                plotter.add_mesh(surface, name="density", color="#e2d6c3",
                                 opacity=state["opacity"], smooth_shading=True, reset_camera=False)
            message = f"{path.name} | TIFF isosurface: {state['density']:.4f} | V: volume / surface; S: save"
        else:
            # 光学透明度仅用于观察累积密度，不声称复现 X 射线透射物理。
            levels = np.linspace(0, density_max, 256)
            # 完整的 256 项查找表必须使用 0–255 alpha；0–1 浮点值会截断为零。
            transfer = np.rint(np.where(levels >= state["density"],
                                       state["opacity"] * 255, 0)).astype(np.uint8)
            plotter.add_volume(grid, name="density", scalars="density", cmap="gray",
                               clim=(0, density_max), opacity=transfer, shade=False, reset_camera=False,
                               scalar_bar_args={"color": "white", "position_y": 0.25})
            message = f"{path.name} | TIFF density volume | V: volume / surface; S: save"
        plotter.add_text(message, name="status", position="upper_left", font_size=10, color="white")
        plotter.render()

    def update(name, value):
        state[name] = value
        redraw()

    def save():
        stem = args.output / f"{args.method}_{args.mode}"
        plotter.screenshot(str(stem) + ".png")
        if args.mode == "gaussians":
            mesh, indices, _ = ellipsoid_mesh(parameters, state["density"], state["ratio"], args.max_count)
            if len(indices):
                mesh.save(str(stem) + ".vtp")
                np.savez_compressed(str(stem) + ".npz", **{name: values[indices] for name, values in parameters.items()})
        Path(str(stem) + ".json").write_text(json.dumps({**state, "input": str(path), "max_count": args.max_count}, indent=2))
        print(f"Saved {stem}")

    redraw()
    plotter.add_slider_widget(lambda value: update("density", value), (0, max(density_max, 1e-6)),
                              value=min(state["density"], density_max), title="Minimum density",
                              pointa=(0.1, 0.15), pointb=(0.85, 0.15), color="white", interaction_event="end")
    plotter.add_slider_widget(lambda value: update("opacity", value), (0, 1), value=state["opacity"],
                              title="Display opacity", pointa=(0.1, 0.07), pointb=(0.85, 0.07), color="white", interaction_event="end")
    if args.mode == "gaussians":
        plotter.add_slider_widget(lambda value: update("ratio", value), (1, args.max_ratio),
                                  value=args.max_ratio, title="Maximum axis ratio",
                                  pointa=(0.1, 0.23), pointb=(0.85, 0.23), color="white", interaction_event="end")
    plotter.add_key_event("s", save)

    def save_current_camera():
        save_camera(camera_destination, capture_pyvista_camera(plotter))

    def reload_camera():
        try:
            apply_pyvista_camera(plotter, load_camera(args.camera or camera_destination))
            plotter.render()
        except (ValueError, OSError) as error:
            print(f'Camera load failed: {error}')

    plotter.add_key_event('c', save_current_camera)
    plotter.add_key_event('l', reload_camera)
    if args.mode in ("tiff", "volume"):
        def toggle_tiff_style():
            state["tiff_style"] = "volume" if state["tiff_style"] == "surface" else "surface"
            redraw()

        plotter.add_key_event("v", toggle_tiff_style)
    if args.mode == "gaussians":
        capped_count = args.max_count or 5000

        def toggle_count():
            args.max_count = 0 if args.max_count else capped_count
            redraw()

        plotter.add_key_event("a", toggle_count)
    plotter.reset_camera()
    apply_pyvista_camera(plotter, camera_config)
    print(f'C: save camera to {camera_destination}; L: reload camera JSON')
    print("Drag sliders; rotate with mouse; press S to save. A graphical desktop is required.")
    plotter.show()


if __name__ == "__main__":
    main()
