import type maplibregl from "maplibre-gl";
import { ZarrLayer } from "@carbonplan/zarr-layer";
import { MIN_ALPHA } from "./highlight";
import { CLASS_COLOR_GLSL, scoreGlsl } from "./classify";
import { dimensionGlsl } from "./colormaps";

/**
 * TESSERA v1.1 embeddings (University of Cambridge) on Source Cooperative,
 * from dClimate's wall-to-wall global run: 128 int8 dimensions per 10 m
 * pixel, 2017-2025, one Zarr group per UTM zone. Each pixel's vector has its
 * own scale (a separate `scales` array), which cosine similarity divides
 * out, so only the raw int8 values are read.
 *
 * Read straight from the S3 bucket behind Source Cooperative, as GeoTessera
 * does: the data.source.coop gateway can throttle the volume a view needs.
 * Path style, since the dotted bucket name breaks virtual-host TLS.
 */
export const TESSERA_BANDS = 128;
// The dimensions the false-color view shows as red, green and blue.
export const TESSERA_RGB_BANDS = [0, 1, 2];
// The colormap's span for one unit-length dimension: about the 1st to 99th
// percentile.
export const TESSERA_DIMENSION_RANGE = 0.2;
const STORE =
  "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate";
const YEAR_ORIGIN = 2017;
const BAND_INDICES = Array.from({ length: TESSERA_BANDS }, (_, i) => i);

const zoneFor = (lng: number) =>
  Math.min(60, Math.max(1, Math.floor((lng + 180) / 6) + 1));

const selector = (year: number) => ({
  time: { selected: year - YEAR_ORIGIN, type: "index" as const },
  band: { selected: BAND_INDICES, type: "index" as const },
});

// The store's fill value is 0, which is also an ordinary value for any one
// dimension, so zarr-layer's NaN for it reads back as 0 here. A pixel with
// no data is all zeros: a vector of length 0.
const component = (i: number) => `(isnan(band_${i}) ? 0.0 : band_${i})`;
const terms = BAND_INDICES.map(
  (i) => `  e[${i}] = ${component(i)};\n  len2 += e[${i}] * e[${i}];`,
).join("\n");

const [r, g, b] = TESSERA_RGB_BANDS.map((i) => `e[${i}]`);

// u_mode: 0 = embeddings as color, 1 = similarity, 2 = classify. In the
// first, u_band picks one dimension for the colormap, or is -1 for RGB.
const customFrag = `
  uniform float u_mode;
  uniform float u_band;
  uniform float u_threshold;
  uniform float u_count;

  float e[${TESSERA_BANDS}];
  float len2 = 0.0;
${terms}
  if (len2 == 0.0) {
    discard;
  }
  float len = sqrt(len2);

  if (u_mode < 0.5) {
    if (u_band < 0.0) {
      vec3 rgb = clamp(vec3(${r}, ${g}, ${b}) / len * 1.5 + 0.5, 0.0, 1.0);
      fragColor = vec4(rgb * opacity, opacity);
    } else {
${dimensionGlsl("e[int(u_band)] / len", TESSERA_DIMENSION_RANGE)}
    }
  } else if (u_count < 0.5) {
    discard;
  } else {
${scoreGlsl(TESSERA_BANDS, "len")}
    if (best < u_threshold) {
      discard;
    }
    if (u_mode < 1.5) {
      float t = clamp((best - u_threshold) / max(1.0 - u_threshold, 1e-3), 0.0, 1.0);
      vec4 c = texture(colormap, vec2(t, 0.5));
      float a = opacity * mix(${MIN_ALPHA}, 1.0, t);
      fragColor = vec4(c.rgb * a, a);
    } else {
${CLASS_COLOR_GLSL}
    }
  }
`;

type Style = {
  clim: [number, number];
  colormap: string[];
  opacity: number;
  minzoom: number;
  customFrag?: string;
  uniforms: Record<string, number | number[] | Float32Array>;
  onLoadingStateChange: (state: { loading: boolean; error?: Error | null }) => void;
};

/**
 * One zarr-layer per UTM zone, created as the view reaches each zone and
 * kept for panning back.
 */
export class TesseraLayers {
  private layers = new Map<number, ZarrLayer>();
  private visible = false;
  private year: number;

  constructor(
    private map: maplibregl.Map,
    private style: (id: string) => Style,
    private beforeId: string,
    private minzoom: number,
    year: number,
  ) {
    this.year = year;
  }

  private id(zone: number) {
    return `tessera-${String(zone).padStart(2, "0")}`;
  }

  /** Add layers for the zones in view and show or hide them all. */
  update(visible: boolean) {
    this.visible = visible;
    if (visible && this.map.getZoom() >= this.minzoom) {
      const bounds = this.map.getBounds();
      const first = zoneFor(bounds.getWest());
      const last = zoneFor(bounds.getEast());
      for (let zone = first; zone <= last; zone++) this.ensure(zone);
    }
    for (const zone of this.layers.keys()) {
      this.map.setLayoutProperty(this.id(zone), "visibility", visible ? "visible" : "none");
    }
  }

  private ensure(zone: number) {
    if (this.layers.has(zone)) return;
    const id = this.id(zone);
    const layer = new ZarrLayer({
      id,
      source: `${STORE}/utm${String(zone).padStart(2, "0")}`,
      variable: "embeddings",
      selector: selector(this.year),
      // 32x32 chunks; group them so a view is dozens of regions, not thousands.
      minRegionSize: 256,
      ...this.style(id),
      customFrag,
    });
    this.layers.set(zone, layer);
    this.map.addLayer(layer, this.beforeId);
  }

  ids(): string[] {
    return [...this.layers.keys()].map((zone) => this.id(zone));
  }

  layerAt(lng: number): ZarrLayer | undefined {
    return this.visible ? this.layers.get(zoneFor(lng)) : undefined;
  }

  setUniforms(uniforms: Record<string, number | number[] | Float32Array>) {
    for (const layer of this.layers.values()) layer.setUniforms(uniforms);
  }

  setColormap(colormap: string[]) {
    for (const layer of this.layers.values()) layer.setColormap(colormap);
  }

  setOpacity(opacity: number) {
    for (const layer of this.layers.values()) layer.setOpacity(opacity);
  }

  setYear(year: number) {
    this.year = year;
    for (const layer of this.layers.values()) layer.setSelector(selector(year));
  }
}

/** Unit-length TESSERA embedding at a point, or null where there is no data. */
export async function readTesseraPoint(
  layer: ZarrLayer,
  lng: number,
  lat: number,
): Promise<number[] | null> {
  const result = await layer.queryData(
    { type: "Point", coordinates: [lng, lat] },
    undefined,
    { includeSpatialCoordinates: false },
  );
  const byBand = (result as Record<string, unknown>)["embeddings"];
  if (!byBand || typeof byBand !== "object") return null;
  const series = Object.values(byBand as Record<string, number[]>);
  if (series.length !== TESSERA_BANDS) return null;
  const raw = series.map((values) => (Number.isFinite(values[0]) ? values[0] : 0));
  const length = Math.hypot(...raw);
  return length > 0 ? raw.map((v) => v / length) : null;
}
