import type maplibregl from "maplibre-gl";
import { ZarrLayer } from "@carbonplan/zarr-layer";

/**
 * TESSERA v1 embeddings (University of Cambridge) on Source Cooperative: 128
 * int8 dimensions per 10 m pixel, 2017-2025, one Zarr group per UTM zone.
 * Each pixel's vector has its own scale (a separate `scales` array), which
 * cosine similarity divides out, so only the raw int8 values are read.
 */
export const TESSERA_BANDS = 128;
const STORE = "https://data.source.coop/tessera/tessera/zarr/v1";
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
  (i) =>
    `  v = ${component(i)};\n  dotq += v * u_q[${i}];\n  len2 += v * v;`,
).join("\n");

// u_mode: 0 = embeddings as color, 1 = similarity.
const customFrag = `
  uniform float u_mode;
  uniform float u_threshold;
  uniform float u_hasQuery;

  float v;
  float dotq = 0.0;
  float len2 = 0.0;
${terms}
  if (len2 == 0.0) {
    discard;
  }
  float len = sqrt(len2);

  if (u_mode < 0.5) {
    vec3 rgb = clamp(vec3(${component(0)}, ${component(1)}, ${component(2)}) / len * 1.5 + 0.5, 0.0, 1.0);
    fragColor = vec4(rgb * opacity, opacity);
  } else if (u_hasQuery < 0.5) {
    discard;
  } else {
    float score = dotq / len;
    if (score < u_threshold) {
      discard;
    }
    float t = (score - u_threshold) / max(1.0 - u_threshold, 1e-3);
    vec4 c = texture(colormap, vec2(clamp(t, 0.0, 1.0), 0.5));
    fragColor = vec4(c.rgb * opacity, opacity);
  }
`;

type Style = {
  clim: [number, number];
  colormap: string[];
  opacity: number;
  minzoom: number;
  customFrag?: string;
  uniforms: Record<string, number | number[]>;
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

  setUniforms(uniforms: Record<string, number | number[]>) {
    for (const layer of this.layers.values()) layer.setUniforms(uniforms);
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
