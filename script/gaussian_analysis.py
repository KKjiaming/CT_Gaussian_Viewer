"""Editable Gaussian model and direct GPU evaluation of its orthogonal slices."""
import json
from pathlib import Path

import numpy as np

from gs_renderer import GaussianRenderer

SUPPORT_SIGMA = 3.0


def slice_gaussian_ids(parameters,axis,coordinate,hidden,minimum_density=0.,maximum_ratio=1e12,maximum_count=0):
    """Gaussians with nonzero slice support, ranked by their conditional peak density."""
    scales = parameters['scales']
    normal_axes = parameters['rotations'][:,axis,:]*scales
    variance = np.sum(normal_axes*normal_axes,axis=1)
    residual = (coordinate-parameters['centers'][:,axis])**2/variance
    peak = parameters['density']*np.exp(-.5*residual)
    ratio = scales.max(axis=1)/scales.min(axis=1)
    mask = (~hidden)&(residual<SUPPORT_SIGMA**2)&(peak>0)&(peak>=minimum_density)&(ratio<=maximum_ratio)
    ids = np.flatnonzero(mask)
    matched = len(ids)
    if maximum_count>0 and matched>maximum_count:
        ids = ids[np.argsort(-peak[ids],kind='stable')[:maximum_count]]
    return ids,matched


def point_contributions(parameters, point, hidden=None):
    """Actual activated density at a world XYZ point, with the slice's 3-sigma support."""
    point = np.asarray(point,dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Expected one finite world XYZ point')
    delta = point-parameters['centers']
    local = np.einsum('ni,nij->nj',delta,parameters['rotations'])/parameters['scales']
    q = np.sum(local*local,axis=1)
    values = parameters['density']*np.exp(-.5*q)
    values[q>SUPPORT_SIGMA**2] = 0
    if hidden is not None:
        values[hidden] = 0
    return values


def pixel_world(axis, row, column, coordinate, bbox, resolution):
    bbox = np.asarray(bbox,dtype=float)
    remaining = [i for i in range(3) if i!=axis]
    point = np.asarray(bbox,dtype=float).mean(axis=0)
    point[axis] = coordinate
    point[remaining] = bbox[0,remaining]+(np.array([column,row])+.5)/resolution*(bbox[1,remaining]-bbox[0,remaining])
    return point


def region_contributions(parameters, axis, coordinate, bbox, resolution, rectangle, hidden=None):
    """Exact per-Gaussian sum over pixel centers in an inclusive (r0,r1,c0,c1) ROI.

    Cull by each Gaussian's conditional support on the slice, then evaluate only
    its intersecting pixels. Contributions use the same cutoff as the GPU slices.
    """
    bbox = np.asarray(bbox,dtype=float)
    r0,r1,c0,c1 = map(int,rectangle)
    if not (0<=r0<=r1<resolution and 0<=c0<=c1<resolution):
        raise ValueError('ROI must be inside the slice raster')
    remaining = [i for i in range(3) if i!=axis]
    axes = parameters['rotations']*parameters['scales'][:,None,:]
    covariance = axes @ axes.transpose(0,2,1)
    variance = covariance[:,axis,axis]
    h = coordinate-parameters['centers'][:,axis]
    residual = h*h/variance
    active = residual<SUPPORT_SIGMA**2
    if hidden is not None:
        active &= ~hidden
    cross = covariance[:,remaining,axis]
    mean = parameters['centers'][:,remaining]+cross*(h/variance)[:,None]
    conditional = covariance[:,remaining,:][:,:,remaining]-cross[:,:,None]*cross[:,None,:]/variance[:,None,None]
    extents = np.sqrt(np.maximum((SUPPORT_SIGMA**2-residual)[:,None]*
                                np.diagonal(conditional,axis1=1,axis2=2),0))
    spacing = (bbox[1,remaining]-bbox[0,remaining])/resolution
    lower = np.ceil((mean-extents-bbox[0,remaining])/spacing-.5).astype(int)
    upper = np.floor((mean+extents-bbox[0,remaining])/spacing-.5).astype(int)
    lower = np.maximum(lower,[c0,r0])
    upper = np.minimum(upper,[c1,r1])
    active &= np.all(lower<=upper,axis=1)
    scores = np.zeros(len(parameters['density']),dtype=float)
    for i in np.flatnonzero(active):
        cols = np.arange(lower[i,0],upper[i,0]+1)
        rows = np.arange(lower[i,1],upper[i,1]+1)
        dx = (bbox[0,remaining[0]]+(cols+.5)*spacing[0]-mean[i,0])[None,:]
        dy = (bbox[0,remaining[1]]+(rows+.5)*spacing[1]-mean[i,1])[:,None]
        a,b,c = conditional[i,0,0],conditional[i,0,1],conditional[i,1,1]
        q = (c*dx*dx-2*b*dx*dy+a*dy*dy)/(a*c-b*b)+residual[i]
        scores[i] = parameters['density'][i]*np.sum(np.where(q<=SUPPORT_SIGMA**2,np.exp(-.5*q),0))
    return scores


def rank_contributions(scores, count=10):
    scores = np.asarray(scores)
    ids = np.flatnonzero(scores>0)
    ids = ids[np.argsort(-scores[ids],kind='stable')][:count]
    total = float(scores.sum())
    return ids, scores[ids], scores[ids]/total if total>0 else np.zeros(len(ids)), total


SLICE_VERTEX = """
#version 330
in vec3 center;
in vec3 axis0;
in vec3 axis1;
in vec3 axis2;
in float density;
uniform vec3 right;
uniform vec3 up;
uniform vec3 normal;
uniform vec3 plane_center;
uniform vec2 half_span;
out vec2 offset;
flat out vec3 inverse_cov;
flat out float weight;
flat out float residual;
const vec2 corners[6] = vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),
                               vec2(-1,-1),vec2(1,1),vec2(-1,1));
void main() {
    vec3 x=vec3(dot(right,axis0),dot(right,axis1),dot(right,axis2));
    vec3 y=vec3(dot(up,axis0),dot(up,axis1),dot(up,axis2));
    vec3 z=vec3(dot(normal,axis0),dot(normal,axis1),dot(normal,axis2));
    float zz=max(dot(z,z),1e-30);
    float h=dot(normal,plane_center-center);
    residual=h*h/zz;
    if(residual>=9.0 || density<=0.0) {
        gl_Position=vec4(2,2,0,1);offset=vec2(0);inverse_cov=vec3(1,0,1);weight=0;
        return;
    }
    float xz=dot(x,z), yz=dot(y,z);
    float a=max(dot(x,x)-xz*xz/zz,1e-20);
    float b=dot(x,y)-xz*yz/zz;
    float c=max(dot(y,y)-yz*yz/zz,1e-20);
    float det=max(a*c-b*b,1e-30);
    inverse_cov=vec3(c,-b,a)/det;
    weight=density*exp(-0.5*residual);
    offset=corners[gl_VertexID]*sqrt((9.0-residual)*vec2(a,c));
    vec3 relative=center-plane_center;
    vec2 mean=vec2(dot(right,relative),dot(up,relative))+vec2(xz,yz)*h/zz;
    gl_Position=vec4((mean+offset)/half_span,0,1);
}
"""

SLICE_FRAGMENT = """
#version 330
in vec2 offset;
flat in vec3 inverse_cov;
flat in float weight;
flat in float residual;
out float value;
void main() {
    float q=inverse_cov.x*offset.x*offset.x+2.0*inverse_cov.y*offset.x*offset.y+
            inverse_cov.z*offset.y*offset.y;
    if(q+residual>9.0 || weight==0.0) discard;
    value=weight*exp(-0.5*max(q,0.0));
}
"""


class EditableGaussians:
    """Keep immutable source indices; display filtering never renumbers original IDs."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.original_ids = parameters.get('original_ids',np.arange(len(parameters['density'])))
        self.hidden = np.zeros(len(parameters['density']), dtype=bool)
        self.selected = None
        self.history = []

    @property
    def density(self):
        return np.where(self.hidden, 0, self.parameters['density'])

    def select(self, original_id):
        original_id = int(original_id)
        if not 0 <= original_id < len(self.hidden):
            raise ValueError('Gaussian ID out of range')
        self.selected = original_id

    def hide_selected(self):
        if self.selected is None or self.hidden[self.selected]:
            return False
        return self.hide_ids([self.selected])

    def hide_ids(self, ids):
        ids = np.unique(np.asarray(ids,dtype=int))
        if np.any(ids<0) or np.any(ids>=len(self.hidden)):
            raise ValueError('Gaussian ID out of range')
        ids = ids[~self.hidden[ids]]
        if not len(ids):
            return False
        self.hidden[ids] = True
        self.history.append(ids)
        return True

    def undo(self):
        if not self.history:
            return False
        self.hidden[self.history.pop()] = False
        return True

    def restore_all(self):
        changed = bool(self.hidden.any())
        self.hidden[:] = False
        self.history.clear()
        return changed

    def selected_parameters(self):
        if self.selected is None:
            return None
        i = self.selected
        scales = self.parameters['scales'][i]
        return dict(original_id=int(self.original_ids[i]), source_row=i, hidden=bool(self.hidden[i]),
                    center=self.parameters['centers'][i].tolist(), scales=scales.tolist(),
                    rotation=self.parameters['rotations'][i].tolist(),
                    density=float(self.parameters['density'][i]),
                    aspect_ratio=float(scales.max()/scales.min()))

    def export(self, destination, source):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        keep = ~self.hidden
        exported = {k:v[keep] for k,v in self.parameters.items()}
        exported['original_ids'] = self.original_ids[keep]
        np.savez_compressed(destination/'edited_parameters.npz',**exported)
        (destination/'edits.json').write_text(json.dumps(dict(source=str(source),
            hidden_ids=self.original_ids[self.hidden].tolist(), hidden_source_rows=np.flatnonzero(self.hidden).tolist(),
            selected=self.selected_parameters()), indent=2))


class AnalysisGPU:
    """One float additive framebuffer for slices; another for analytic projections."""
    def __init__(self, ctx, parameters, bbox):
        self.ctx = ctx
        self.bbox = np.asarray(bbox, dtype=float)
        if self.bbox.shape != (2,3) or not np.isfinite(self.bbox).all() or np.any(self.bbox[1]<=self.bbox[0]):
            raise ValueError('Expected finite bounding box with lower < upper')
        self.projector = GaussianRenderer(ctx, parameters)
        self.program = ctx.program(vertex_shader=SLICE_VERTEX, fragment_shader=SLICE_FRAGMENT)
        self.vao = ctx.vertex_array(self.program, [(self.projector.buffer, '3f 3f 3f 3f 1f /i',
                                                  'center','axis0','axis1','axis2','density')])
        self.texture = self.fbo = self.output = None
        self.size = None

    def _resize(self, size):
        if size == self.size:
            return
        for resource in (self.fbo, self.texture, self.output):
            if resource is not None:
                resource.release()
        self.texture = self.ctx.texture(size, 1, dtype='f4')
        self.fbo = self.ctx.framebuffer([self.texture])
        self.output = self.ctx.simple_framebuffer(size, components=3)
        self.size = size

    def update_density(self, density):
        self.projector.update_density(density)

    def slice(self, axis, coordinate, resolution):
        import moderngl
        if axis not in (0,1,2) or resolution < 2 or not np.isfinite(coordinate):
            raise ValueError('Invalid slice axis, resolution or coordinate')
        self._resize((resolution, resolution))
        remaining = [i for i in range(3) if i != axis]
        right, up = np.eye(3)[remaining]
        normal = np.eye(3)[axis]
        center = self.bbox.mean(axis=0)
        center[axis] = coordinate
        half_span = (self.bbox[1]-self.bbox[0])[remaining]/2
        for name,value in dict(right=tuple(right),up=tuple(up),normal=tuple(normal),
                               plane_center=tuple(center),half_span=tuple(half_span)).items():
            self.program[name].value = value
        self.fbo.use()
        self.ctx.viewport = (0,0,resolution,resolution)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.fbo.clear()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.vao.render(vertices=6, instances=self.projector.count)
        self.ctx.disable(moderngl.BLEND)
        return np.frombuffer(self.fbo.read(components=1,dtype='f4'),dtype='f4').reshape(resolution,resolution).copy()

    def projection(self, camera, resolution):
        from gs_renderer import configure_camera
        size = (resolution,resolution) if np.isscalar(resolution) else tuple(resolution)
        self._resize(size)
        target = self.bbox.mean(axis=0)
        state = dict(yaw=.7,pitch=.4,roll=0.,distance=3.,span=1.3)
        configure_camera(camera,state,target)
        self.projector.render(self.size,self.output,target,state['yaw'],state['pitch'],
                              state['span'],1.,0.,1e12,roll=state['roll'])
        return np.frombuffer(self.projector.fbo.read(components=1,dtype='f4'),dtype='f4').reshape(self.size[::-1]).copy()

    def release(self):
        for item in (self.vao,self.program,self.fbo,self.texture,self.output):
            if item is not None:
                item.release()
        self.projector.release()
