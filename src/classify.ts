/**
 * Classification: each class has a query vector built the same way as the
 * similarity search, and a pixel takes the class whose vector scores it
 * highest. The vectors travel to the shader packed into one uniform array,
 * `u_q`, MAX_CLASSES slots of `dims` floats each; similarity mode is the
 * one-class case. Every slot costs fragment uniform vectors whether or not
 * it is used, which caps the number of classes.
 */
export const MAX_CLASSES = 6;

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
