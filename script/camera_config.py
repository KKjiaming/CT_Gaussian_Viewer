"""Shared world-coordinate camera configuration for all viewer modes."""
import json
from pathlib import Path

import numpy as np


def aligned_camera(axis, reverse, bbox, slice_positions, viewport_size):
    """Match slice screen axes: X->(Y,Z), Y->(X,Z), Z->(X,Y)."""
    if axis not in (0,1,2):
        raise ValueError('Camera axis must be 0, 1 or 2')
    bbox = np.asarray(bbox,dtype=float)
    remaining = [i for i in range(3) if i!=axis]
    extent = bbox[1]-bbox[0]
    target = bbox.mean(axis=0)
    target[axis] = slice_positions[axis]
    direction = np.zeros(3)
    direction[axis] = (1.,-1.,1.)[axis]*(-1 if reverse else 1)
    up = np.eye(3)[remaining[1]]
    distance = max(float(np.linalg.norm(extent))*2,1.)
    width,height = viewport_size
    aspect = max(width,1)/max(height,1)
    span = max(extent[remaining[1]]/.65,extent[remaining[0]]/(.9*aspect))/2
    return validate_camera(dict(position=(target+direction*distance).tolist(),target=target.tolist(),
                                up=up.tolist(),projection='orthographic',parallel_scale=float(span)))


def validate_camera(data):
    allowed = {'position', 'target', 'up', 'projection', 'parallel_scale', 'view_angle',
               'clipping_range', 'window_size'}
    unknown = set(data)-allowed
    if unknown:
        raise ValueError(f'Unknown camera fields: {sorted(unknown)}')
    result = dict(data)
    for name in ('position', 'target', 'up'):
        if name in result:
            a = np.asarray(result[name], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f'{name} must contain three finite numbers')
            result[name] = a.tolist()
    if 'up' in result and np.linalg.norm(result['up']) < 1e-10:
        raise ValueError('up must be nonzero')
    if 'position' in result and 'target' in result:
        direction = np.array(result['position'])-result['target']
        if np.linalg.norm(direction) < 1e-10:
            raise ValueError('position and target must differ')
        if 'up' in result and np.linalg.norm(np.cross(direction/np.linalg.norm(direction),
                                                      np.array(result['up'])/np.linalg.norm(result['up']))) < 1e-8:
            raise ValueError('up must not be parallel to the viewing direction')
    if result.get('projection', 'orthographic') not in ('orthographic', 'perspective'):
        raise ValueError('projection must be orthographic or perspective')
    for name in ('parallel_scale', 'view_angle'):
        if name in result:
            value = float(result[name])
            if not np.isfinite(value) or value <= 0 or (name == 'view_angle' and value >= 180):
                raise ValueError(f'Invalid {name}')
            result[name] = value
    for name in ('clipping_range', 'window_size'):
        if name in result:
            a = np.asarray(result[name], dtype=float)
            if a.shape != (2,) or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f'{name} must contain two positive values')
            if name == 'clipping_range' and a[1] <= a[0]:
                raise ValueError('clipping_range requires near < far')
            if name == 'window_size' and np.any(a != np.floor(a)):
                raise ValueError('window_size must contain integers')
            result[name] = a.astype(int).tolist() if name == 'window_size' else a.tolist()
    return result


def load_camera(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Camera JSON must be an object')
    return validate_camera(data)


def save_camera(path, data):
    data = validate_camera(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')
    print(f'Saved camera: {path}')


def add_camera_arguments(parser):
    parser.add_argument('--camera', type=Path, help='相机 JSON；命令行参数覆盖文件参数')
    parser.add_argument('--camera-save', type=Path, help='C 键保存相机的路径')
    for name in ('position', 'target', 'up'):
        parser.add_argument(f'--camera-{name}', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--projection', choices=['orthographic', 'perspective'])
    parser.add_argument('--parallel-scale', type=float, help='正交相机画面垂直半高度（世界坐标）')
    parser.add_argument('--view-angle', type=float, help='透视相机垂直视场角（度）；仅高斯球/TIFF 模式')
    parser.add_argument('--clipping-range', type=float, nargs=2, metavar=('NEAR', 'FAR'),
                        help='仅高斯球/TIFF；直接渲染沿完整射线积分')
    parser.add_argument('--window-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))


def camera_from_args(args):
    data = load_camera(args.camera) if args.camera else {}
    for name in ('position', 'target', 'up', 'projection', 'parallel_scale', 'view_angle',
                 'clipping_range', 'window_size'):
        value = getattr(args, 'camera_'+name if name in ('position', 'target', 'up') else name)
        if value is not None:
            data[name] = value
    return validate_camera(data)


def apply_pyvista_camera(plotter, config):
    camera = plotter.camera
    merged = validate_camera({**capture_pyvista_camera(plotter), **config})
    camera.position = merged['position']
    camera.focal_point = merged['target']
    camera.up = merged['up']
    camera.parallel_projection = merged['projection'] == 'orthographic'
    camera.parallel_scale = merged['parallel_scale']
    camera.view_angle = merged['view_angle']
    if 'clipping_range' not in config:
        plotter.reset_camera_clipping_range()
    else:
        camera.clipping_range = merged['clipping_range']
    if 'window_size' in config and tuple(plotter.window_size)!=tuple(config['window_size']):
        plotter.window_size = merged['window_size']


def capture_pyvista_camera(plotter):
    c = plotter.camera
    return dict(position=list(c.position), target=list(c.focal_point), up=list(c.up),
                projection='orthographic' if c.parallel_projection else 'perspective',
                parallel_scale=float(c.parallel_scale), view_angle=float(c.view_angle),
                clipping_range=list(c.clipping_range), window_size=list(plotter.window_size))
