"""WGSL sources for the EasyGPU example programs.

Faithful ports of the ``webgpu-samples`` shaders (``shaders/`` and each
``sample/*.wgsl``). EasyGPU inlines WGSL as Python strings because the library
has zero runtime dependencies — there is no file loader. Keeping them in one
place mirrors the samples' shared ``shaders/`` directory.
"""

from __future__ import annotations

# --- shaders/basic.vert.wgsl + shaders/vertexPositionColor.frag.wgsl --------

BASIC_VERT = """
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
};

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(@location(0) position : vec4f, @location(1) uv : vec2f) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix * position;
  output.fragUV = uv;
  output.fragPosition = 0.5 * (position + vec4f(1.0));
  return output;
}
"""

VERTEX_POSITION_COLOR_FRAG = """
@fragment
fn main(@location(0) fragUV : vec2f, @location(1) fragPosition : vec4f) -> @location(0) vec4f {
  return vec4f(fragPosition.rgb, 1.0);
}
"""

# --- sample/instancedCube/instanced.vert.wgsl -------------------------------

INSTANCED_VERT = """
struct InstanceInput {
  @location(0) position : vec4f,
  @location(1) uv : vec2f,
};

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
  @location(1) fragPosition: vec4f,
};

struct Uniforms {
  modelViewProjectionMatrix : array<mat4x4f, 16>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(input : InstanceInput, @builtin(instance_index) instanceIndex : u32) -> VertexOutput {
  var output : VertexOutput;
  output.Position = uniforms.modelViewProjectionMatrix[instanceIndex] * input.position;
  output.fragUV = input.uv;
  output.fragPosition = 0.5 * (input.position + vec4f(1.0));
  return output;
}
"""

# --- sample/texturedCube/sampleTextureMixColor.frag.wgsl --------------------

TEXTURED_CUBE_FRAG = """
@group(0) @binding(1) var ourSampler: sampler;

@group(0) @binding(2) var ourTexture: texture_2d<f32>;

@fragment
fn main(@location(0) fragUV : vec2f, @location(1) fragPosition : vec4f) -> @location(0) vec4f {
  return (textureSample(ourTexture, ourSampler, fragUV) * fragPosition);
}
"""

# --- sample/cameras/cube.wgsl ------------------------------------------------

CAMERA_CUBE = """
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vertex_main(
  @location(0) position : vec4f,
  @location(1) uv : vec2f
) -> VertexOutput {
  return VertexOutput(uniforms.modelViewProjectionMatrix * position, uv);
}

@fragment
fn fragment_main(@location(0) fragUV: vec2f) -> @location(0) vec4f {
  return textureSample(myTexture, mySampler, fragUV);
}
"""

# --- sample/points/*.wgsl -----------------------------------------------------

POINT_DISTANCE_VERT = """
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
"""

POINT_FIXED_VERT = """
struct Vertex {
  @location(0) position: vec4f,
};

struct Uniforms {
  matrix: mat4x4f,
  resolution: vec2f,
  size: f32,
};

struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
    vert: Vertex,
    @builtin(vertex_index) vNdx: u32,
) -> VSOutput {
  let points = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f(-1,  1),
    vec2f( 1, -1),
    vec2f( 1,  1),
  );
  var vsOut: VSOutput;
  let pos = points[vNdx];
  let clipPos = uni.matrix * vert.position;
  let pointPos = vec4f(pos * uni.size / uni.resolution * clipPos.w, 0, 0);
  vsOut.position = clipPos + pointPos;
  vsOut.texcoord = pos * 0.5 + 0.5;
  return vsOut;
}
"""

POINT_ORANGE_FRAG = """
@fragment fn fs() -> @location(0) vec4f {
  return vec4f(1, 0.5, 0.2, 1);
}
"""

POINT_TEXTURED_FRAG = """
struct VSOutput {
  @location(0) texcoord: vec2f,
};

@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var t: texture_2d<f32>;

@fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
  let color = textureSample(t, s, vsOut.texcoord);
  if (color.a < 0.1) {
    discard;
  }
  return color;
}
"""

# --- sample/reversedZ/vertex.wgsl + fragment.wgsl ----------------------------

REVERSED_Z_VERT = """
struct Uniforms {
  modelMatrix : array<mat4x4f, 5>,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<uniform> camera : Camera;

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragColor : vec4f,
}

@vertex
fn main(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) position : vec4f,
  @location(1) color : vec4f
) -> VertexOutput {
  var output : VertexOutput;
  output.Position = camera.viewProjectionMatrix * uniforms.modelMatrix[instanceIdx] * position;
  output.fragColor = color;
  return output;
}
"""

REVERSED_Z_FRAG = """
@fragment
fn main(
  @location(0) fragColor: vec4f
) -> @location(0) vec4f {
  return fragColor;
}
"""

# --- sample/blending/texturedQuad.wgsl -----------------------------------------

TEXTURED_QUAD = """
struct OurVertexShaderOutput {
  @builtin(position) position: vec4f,
  @location(0) texcoord: vec2f,
};

struct Uniforms {
  matrix: mat4x4f,
};

@group(0) @binding(2) var<uniform> uni: Uniforms;

@vertex fn vs(
  @builtin(vertex_index) vertexIndex : u32
) -> OurVertexShaderOutput {
  let pos = array(
    vec2f( 0.0,  0.0),  // center
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 0.0,  1.0),  // center, top
    vec2f( 0.0,  1.0),  // center, top
    vec2f( 1.0,  0.0),  // right, center
    vec2f( 1.0,  1.0),  // right, top
  );
  var vsOutput: OurVertexShaderOutput;
  let xy = pos[vertexIndex];
  vsOutput.position = uni.matrix * vec4f(xy, 0.0, 1.0);
  vsOutput.texcoord = xy;
  return vsOutput;
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(fsInput: OurVertexShaderOutput) -> @location(0) vec4f {
  return textureSample(ourTexture, ourSampler, fsInput.texcoord);
}
"""

# --- wireframe: minimal solid + edge-line shaders ------------------------------

CUBE_SOLID_VERT = """
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(@location(0) position : vec3f) -> @builtin(position) vec4f {
  return uniforms.modelViewProjectionMatrix * vec4f(position, 1.0);
}
"""

CUBE_SOLID_FRAG = """
@fragment
fn main() -> @location(0) vec4f {
  return vec4f(0.4, 0.6, 0.9, 1.0);
}
"""

CUBE_WIRE_VERT = """
struct Uniforms {
  modelViewProjectionMatrix : mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(@location(0) position : vec3f) -> @builtin(position) vec4f {
  return uniforms.modelViewProjectionMatrix * vec4f(position, 1.0);
}
"""

CUBE_WIRE_FRAG = """
@fragment
fn main() -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 1.0, 1.0);
}
"""

# --- alpha-to-coverage: reuses TEXTURED_QUAD -----------------------------------

__all__ = [
    "BASIC_VERT",
    "CAMERA_CUBE",
    "CUBE_SOLID_FRAG",
    "CUBE_SOLID_VERT",
    "CUBE_WIRE_FRAG",
    "CUBE_WIRE_VERT",
    "INSTANCED_VERT",
    "POINT_DISTANCE_VERT",
    "POINT_FIXED_VERT",
    "POINT_ORANGE_FRAG",
    "POINT_TEXTURED_FRAG",
    "REVERSED_Z_FRAG",
    "REVERSED_Z_VERT",
    "TEXTURED_CUBE_FRAG",
    "TEXTURED_QUAD",
    "VERTEX_POSITION_COLOR_FRAG",
]
