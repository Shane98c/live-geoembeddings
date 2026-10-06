/**
 * Classification: each class has a query vector built the same way as the
 * similarity search, and a pixel takes the class whose vector scores it
 * highest. The vectors travel to the shader packed into one uniform array,
 * `u_q`, MAX_CLASSES slots of `dims` floats each; similarity mode is the
 * one-class case. Every slot costs fragment uniform vectors whether or not
 * it is used, so the number of classes is sized to the device's budget.
 */
const PREFERRED_CLASSES = 6;
// The widest embedding (TESSERA) sets the slot size; every layer's shader
// must compile. The reserve covers the layers' other uniforms.
const LARGEST_EMBEDDING = 128;
const RESERVED_VECTORS = 48;
// The WebGL2 minimum, assumed when the budget cannot be read.
const MIN_BUDGET = 224;

function fragmentUniformBudget(): number {
  if (typeof document === "undefined") return MIN_BUDGET;
  const gl = document.createElement("canvas").getContext("webgl2");
  const budget = gl?.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);
  gl?.getExtension("WEBGL_lose_context")?.loseContext();
  return typeof budget === "number" && budget > 0 ? budget : MIN_BUDGET;
}

export const MAX_CLASSES = Math.max(
  1,
  Math.min(
    PREFERRED_CLASSES,
    Math.floor((fragmentUniformBudget() - RESERVED_VECTORS) / LARGEST_EMBEDDING),
  ),
);

// Default class colors: saturated hues that stay apart over imagery.
export const PALETTE = [
  "#2ca02c",
  "#1f77b4",
  "#ff7f0e",
  "#d62728",
  "#9467bd",
  "#e7e71a",
];

export type ClassQuery = { vector: number[]; color: string };

export function hexToRgb(hex: string): [number, number, number] {
  return [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16) / 255) as [
    number,
    number,
    number,
  ];
}

/** The class vectors and colors packed for the shader. */
export function classUniforms(
  dims: number,
  queries: ClassQuery[],
): { u_count: number; u_q: Float32Array; u_colors: Float32Array } {
  const u_q = new Float32Array(MAX_CLASSES * dims);
  const u_colors = new Float32Array(MAX_CLASSES * 3);
  const used = queries.slice(0, MAX_CLASSES);
  used.forEach(({ vector, color }, k) => {
    u_q.set(vector, k * dims);
    u_colors.set(hexToRgb(color), k * 3);
  });
  return { u_count: used.length, u_q, u_colors };
}

/**
 * GLSL that scores the pixel embedding `e[dims]` against every class,
 * leaving the best class and score in `bestK` and `best`. `scale` divides
 * the dot product (TESSERA vectors are not unit length).
 */
export function scoreGlsl(dims: number, scale = "1.0"): string {
  return `
  int count = int(u_count);
  float best = -1e9;
  int bestK = 0;
  for (int k = 0; k < ${MAX_CLASSES}; k++) {
    if (k >= count) break;
    float s = 0.0;
    for (int i = 0; i < ${dims}; i++) {
      s += e[i] * u_q[k * ${dims} + i];
    }
    s /= ${scale};
    if (s > best) {
      best = s;
      bestK = k;
    }
  }`;
}

/** GLSL that draws the pixel in the best class's color. */
export const CLASS_COLOR_GLSL = `
    vec3 c = vec3(u_colors[bestK * 3], u_colors[bestK * 3 + 1], u_colors[bestK * 3 + 2]);
    fragColor = vec4(c * opacity, opacity);`;
