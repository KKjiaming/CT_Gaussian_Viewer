"""Direct orthographic CT Gaussian projection: GPU splats, no voxel grid.

Each primitive contributes the analytic line integral of
rho * exp(-0.5 * (x-mu)^T Sigma^-1 (x-mu)). Projected tails are
truncated at 3 sigma. Additive optical depth is displayed as 1-exp(-exposure*tau).
This is a density projection, rather than RGB spherical-harmonic 3DGS.
"""

import time
from pathlib import Path

import numpy as np

VERTEX = """
#version 330
in vec3 center;
in vec3 axis0;
in vec3 axis1;
in vec3 axis2;
in float density;
uniform vec3 right;
uniform vec3 up;
uniform vec3 target;
uniform vec2 half_span;
uniform float minimum_density;
uniform float maximum_ratio;
out vec2 offset;
flat out vec3 inverse_cov;
flat out float weight;
const vec2 corners[6] = vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),
                               vec2(-1,-1),vec2(1,1),vec2(-1,1));
void main() {
    vec3 lengths = vec3(length(axis0),length(axis1),length(axis2));
    if (density < minimum_density || max(max(lengths.x,lengths.y),lengths.z) /
        min(min(lengths.x,lengths.y),lengths.z) > maximum_ratio) {
        gl_Position = vec4(2,2,0,1); offset=vec2(0); inverse_cov=vec3(1,0,1);
        weight=0; return;
    }
    vec3 x = vec3(dot(right,axis0),dot(right,axis1),dot(right,axis2));
    vec3 y = vec3(dot(up,axis0),dot(up,axis1),dot(up,axis2));
    float a=dot(x,x), b=dot(x,y), c=dot(y,y);
    float det=max(a*c-b*b,1e-30);
    inverse_cov=vec3(c,-b,a)/det;
    float determinant3=abs(dot(axis0,cross(axis1,axis2)));
    weight=density*2.50662827463*determinant3/sqrt(det);
    offset=corners[gl_VertexID]*3.0*sqrt(vec2(a,c));
    vec3 relative=center-target;
    vec2 position=vec2(dot(right,relative),dot(up,relative))+offset;
    gl_Position=vec4(position/half_span,0,1);
}
"""

FRAGMENT = """
#version 330
in vec2 offset;
flat in vec3 inverse_cov;
flat in float weight;
out float optical_depth;
void main() {
    float q=inverse_cov.x*offset.x*offset.x+
            2.0*inverse_cov.y*offset.x*offset.y+inverse_cov.z*offset.y*offset.y;
    if(q>9.0 || weight==0.0) discard;
    optical_depth=weight*exp(-0.5*max(q,0.0));
}
"""

DISPLAY_VERTEX = """
#version 330
out vec2 uv;
void main() {
    vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
    uv=p; gl_Position=vec4(p*2.0-1.0,0,1);
}
"""

DISPLAY_FRAGMENT = """
#version 330
uniform sampler2D density_image;
uniform float exposure;
uniform bool transmission;
in vec2 uv;
out vec4 color;
void main() {
    float t=exp(-exposure*max(texture(density_image,uv).r,0.0));
    float value=transmission?t:1.0-t;
    color=vec4(vec3(value),1);
}
"""


def camera_basis(yaw, pitch, roll=0.0):
    """Return orthonormal world-XYZ (screen_right, screen_up) vectors.

    yaw, pitch, and roll are radians. cross(right, up) points from target toward
    the camera. Yaw rotates about world Z; pitch controls elevation. Each return
    value is a NumPy (3,) vector; roll rotates the screen basis about its normal.
    """
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    direction = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    up = np.cross(direction, right)
    return np.cos(roll) * right + np.sin(roll) * up, -np.sin(roll) * right + np.cos(roll) * up


def configure_camera(config, state, target):
    from camera_config import validate_camera

    right, up = camera_basis(state["yaw"], state["pitch"], state.get("roll", 0.0))
    direction = np.cross(right, up)
    defaults = dict(
        position=(target + direction * state["distance"]).tolist(),
        target=target.tolist(),
        up=up.tolist(),
        projection="orthographic",
        parallel_scale=state["span"],
    )
    camera = validate_camera({**defaults, **config})
    if camera["projection"] != "orthographic":
        raise ValueError("Direct GS projection supports orthographic cameras only")
    direction = np.array(camera["position"]) - camera["target"]
    distance = float(np.linalg.norm(direction))
    direction /= distance
    yaw = float(np.arctan2(direction[1], direction[0]))
    pitch = float(np.arcsin(np.clip(direction[2], -1, 1)))
    right, up = camera_basis(yaw, pitch)
    desired_up = np.asarray(camera["up"], dtype=float)
    roll = float(np.arctan2(-np.dot(desired_up, right), np.dot(desired_up, up)))
    target[:] = camera["target"]
    state.update(yaw=yaw, pitch=pitch, roll=roll, distance=distance, span=camera["parallel_scale"])


class GaussianRenderer:
    """Shared window/offscreen rendering implementation, all Gaussians resident on GPU."""

    def __init__(self, ctx, parameters):
        self.ctx = ctx
        self.count = len(parameters["density"])
        # The rotation matrix columns are the principal axes.
        axes = parameters["rotations"] * parameters["scales"][:, None, :]
        self.packed = np.column_stack(
            (parameters["centers"], axes.transpose(0, 2, 1).reshape(-1, 9), parameters["density"])
        ).astype("f4")
        self.buffer = ctx.buffer(self.packed.tobytes())
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.vao = ctx.vertex_array(
            self.program,
            [(self.buffer, "3f 3f 3f 3f 1f /i", "center", "axis0", "axis1", "axis2", "density")],
        )
        self.display = ctx.program(vertex_shader=DISPLAY_VERTEX, fragment_shader=DISPLAY_FRAGMENT)
        self.screen_vao = ctx.vertex_array(self.display, [])
        self.size = None
        self.texture = self.fbo = None

    def update_density(self, density):
        density = np.asarray(density)
        if density.shape != (self.count,) or not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("Expected one finite nonnegative density per Gaussian")
        self.packed[:, -1] = density
        self.buffer.write(self.packed.tobytes())

    def render(
        self,
        size,
        output,
        target,
        yaw,
        pitch,
        span,
        exposure,
        minimum_density,
        maximum_ratio,
        transmission=False,
        roll=0.0,
    ):
        """Draw an orthographic Gaussian projection into a caller-owned framebuffer.

        Args:
            size: (width, height) in pixels, matching output dimensions.
            output: ModernGL framebuffer receiving the exposure-mapped image.
            target: (3,) world-XYZ camera target, in the same units as centers.
            yaw: Camera azimuth in radians.
            pitch: Camera elevation in radians.
            span: Vertical half-span of the view in world units.
            exposure: Positive multiplier used in exponential intensity mapping.
            minimum_density: Minimum activated peak density to render.
            maximum_ratio: Maximum principal-axis scale ratio to render.
            transmission: Display exp(-exposure * integral) instead of its complement.
            roll: Rotation of the screen basis in radians.

        Returns:
            None. Writes the display image to output; self.fbo retains the raw
            single-channel density line integral with a bottom-row image origin.
            The caller must make this renderer's OpenGL context current.
        """
        import moderngl

        if self.size != size:
            if self.fbo is not None:
                self.fbo.release()
                self.texture.release()
            self.texture = self.ctx.texture(size, 1, dtype="f4")
            self.fbo = self.ctx.framebuffer([self.texture])
            self.size = size
        right, up = camera_basis(yaw, pitch, roll)
        for name, value in dict(
            right=tuple(right),
            up=tuple(up),
            target=tuple(target),
            half_span=(span * size[0] / size[1], span),
            minimum_density=minimum_density,
            maximum_ratio=maximum_ratio,
        ).items():
            self.program[name].value = value
        self.ctx.viewport = (0, 0, *size)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.fbo.use()
        self.fbo.clear()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.vao.render(vertices=6, instances=self.count)
        self.ctx.disable(moderngl.BLEND)
        output.use()
        self.texture.use(location=0)
        self.display["density_image"].value = 0
        self.display["exposure"].value = exposure
        self.display["transmission"].value = transmission
        self.screen_vao.render(vertices=3)

    def release(self):
        for item in (
            self.vao,
            self.buffer,
            self.program,
            self.screen_vao,
            self.display,
            self.fbo,
            self.texture,
        ):
            if item is not None:
                item.release()


def run(
    parameters,
    path,
    output_dir,
    minimum_density=0.0,
    maximum_ratio=1e9,
    exposure=2.0,
    camera_config=None,
    camera_destination=None,
    camera_source=None,
):
    import glfw
    import moderngl
    from PIL import Image

    from camera_config import load_camera, save_camera

    camera_config = camera_config or {}
    camera_destination = camera_destination or Path(output_dir) / "camera.json"

    glfw.set_error_callback(lambda code, text: print(f"GLFW {code}: {text.decode()}"))
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; a working graphical session is required")
    window = None
    renderer = None
    ctx = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(
            *camera_config.get("window_size", (1100, 850)), "CT GS direct rendering", None, None
        )
        if not window:
            raise RuntimeError("Could not create an OpenGL 3.3 window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        ctx = moderngl.create_context(require=330)
        renderer = GaussianRenderer(ctx, parameters)
        centers = parameters["centers"]
        low, high = np.quantile(centers, [0.01, 0.99], axis=0)
        target = (low + high) / 2
        initial_span = max(float((high - low).max()) * 0.65, 0.1)
        state = dict(
            yaw=0.7,
            pitch=0.4,
            roll=0.0,
            distance=3.0,
            span=initial_span,
            exposure=exposure,
            transmission=False,
            screenshot=False,
            filtered=False,
            last=None,
        )
        configure_camera(camera_config, state, target)
        initial_camera = {
            name: state[name] for name in ("yaw", "pitch", "roll", "distance", "span")
        }
        initial_target = target.copy()

        def capture_camera():
            right, up = camera_basis(state["yaw"], state["pitch"], state["roll"])
            return dict(
                position=(target + np.cross(right, up) * state["distance"]).tolist(),
                target=target.tolist(),
                up=up.tolist(),
                projection="orthographic",
                parallel_scale=state["span"],
                window_size=list(glfw.get_window_size(window)),
            )

        def scroll(win, dx, dy):
            state["span"] = float(np.clip(state["span"] * np.exp(-dy * 0.1), 1e-4, 1e4))

        def cursor(win, x, y):
            previous = state["last"]
            state["last"] = (x, y)
            if previous is None:
                return
            dx, dy = x - previous[0], y - previous[1]
            if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                state["yaw"] -= dx * 0.006
                state["pitch"] = float(np.clip(state["pitch"] + dy * 0.006, -1.55, 1.55))
            if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                right, up = camera_basis(state["yaw"], state["pitch"], state["roll"])
                height = max(glfw.get_window_size(win)[1], 1)
                target[:] += (-dx * right + dy * up) * 2 * state["span"] / height

        def key(win, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_UP:
                state["exposure"] = min(state["exposure"] * 1.2, 1e5)
            elif key == glfw.KEY_DOWN:
                state["exposure"] = max(state["exposure"] / 1.2, 1e-5)
            elif key == glfw.KEY_T and action == glfw.PRESS:
                state["transmission"] = not state["transmission"]
            elif key == glfw.KEY_F and action == glfw.PRESS:
                state["filtered"] = not state["filtered"]
            elif key == glfw.KEY_S:
                state["screenshot"] = True
            elif key == glfw.KEY_R:
                state.update(**initial_camera, exposure=exposure)
                target[:] = initial_target
            elif key == glfw.KEY_C and action == glfw.PRESS:
                save_camera(camera_destination, capture_camera())
            elif key == glfw.KEY_L and action == glfw.PRESS:
                try:
                    config = load_camera(camera_source or camera_destination)
                    configure_camera(config, state, target)
                    if "window_size" in config:
                        glfw.set_window_size(win, *config["window_size"])
                except (ValueError, OSError) as error:
                    print(f"Camera load failed: {error}")

        glfw.set_scroll_callback(window, scroll)
        glfw.set_cursor_pos_callback(window, cursor)
        glfw.set_key_callback(window, key)
        print(f"GPU: {ctx.info['GL_RENDERER']}; loaded {renderer.count} Gaussians from {path}")
        print(
            "Left drag: orbit; right drag: pan; wheel: zoom; Up/Down: exposure; "
            "T: transmission; F: configured filters; R: reset; S: screenshot; "
            "C: save camera; L: reload camera; Esc: quit"
        )
        if "clipping_range" in camera_config:
            print(
                "Direct GS rendering integrates complete rays; clipping_range applies only to mesh/TIFF views."
            )
        stamp, frames = time.perf_counter(), 0
        while not glfw.window_should_close(window):
            glfw.poll_events()
            size = glfw.get_framebuffer_size(window)
            if min(size) == 0:
                glfw.wait_events_timeout(0.1)
                continue
            renderer.render(
                size,
                ctx.screen,
                target,
                state["yaw"],
                state["pitch"],
                state["span"],
                state["exposure"],
                minimum_density if state["filtered"] else 0.0,
                maximum_ratio if state["filtered"] else 1e9,
                state["transmission"],
                roll=state["roll"],
            )
            if state["screenshot"]:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                destination = output_dir / f"gs_direct_{time.time_ns()}.png"
                Image.frombytes("RGB", size, ctx.screen.read(components=3, alignment=1)).transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                ).save(destination)
                print(f"Saved {destination}")
                state["screenshot"] = False
            glfw.swap_buffers(window)
            frames += 1
            now = time.perf_counter()
            if now - stamp >= 0.5:
                glfw.set_window_title(
                    window,
                    f"CT GS | {Path(path).name} | {renderer.count:,} Gaussians | "
                    f"{frames / (now - stamp):.1f} FPS | exposure {state['exposure']:.2f} | "
                    f"{'filtered' if state['filtered'] else 'all'} | "
                    f"{'transmission' if state['transmission'] else 'density projection'}",
                )
                stamp, frames = now, 0
    finally:
        if renderer is not None:
            renderer.release()
        if ctx is not None:
            ctx.release()
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
