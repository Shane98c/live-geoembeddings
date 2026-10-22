// Colormaps for showing one embedding dimension. Values are signed and
// centered on 0; the red-blue map shows the sign directly.
const STOPS = {
  greyscale: { label: "Greyscale", stops: ["#000000", "#ffffff"] },
  viridis: {
    label: "Viridis",
    stops: ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"],
  },
  magma: {
    label: "Magma",
    stops: ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55964", "#fb8761", "#fec287", "#fcfdbf"],
  },
  redblue: {
    label: "Blue to red",
    stops: ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"],
  },
};
export type ColormapName = keyof typeof STOPS;

const hex = (v: number) => Math.round(v).toString(16).padStart(2, "0");
const channels = (c: string) =>
  [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16));

/**
 * The stops resampled to many evenly spaced colors. The shader reads the
 * colormap as a texture whose texel centers sit at (i + 0.5) / n, so with
 * few stops its ramp would drift from a CSS gradient of the same stops.
 */
function resample(stops: string[], n = 256): string[] {
  return Array.from({ length: n }, (_, i) => {
    const x = (i / (n - 1)) * (stops.length - 1);
    const k = Math.min(Math.floor(x), stops.length - 2);
    const f = x - k;
    const [a, b] = [channels(stops[k]), channels(stops[k + 1])];
    return `#${a.map((v, j) => hex(v + (b[j] - v) * f)).join("")}`;
  });
}

export const DIMENSION_COLORMAPS = Object.fromEntries(
  Object.entries(STOPS).map(([name, { label, stops }]) => [
    name,
    { label, stops, texels: resample(stops) },
  ]),
) as Record<ColormapName, { label: string; stops: string[]; texels: string[] }>;

/** GLSL that draws `value`, one dimension, on the colormap over ±range. */
export const dimensionGlsl = (value: string, range: number) => `
      float t = clamp(${value} / ${(2 * range).toFixed(3)} + 0.5, 0.0, 1.0);
      fragColor = vec4(texture(colormap, vec2(t, 0.5)).rgb * opacity, opacity);`;
