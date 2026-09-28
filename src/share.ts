import { NUM_BANDS } from "./embeddings";
import { TESSERA_BANDS } from "./tessera";

export type Dataset = "aef" | "tessera";

/** A similarity search small enough to travel in a link. */
export interface SharedSearch {
  center: [number, number];
  zoom: number;
  threshold: number;
  counterWeight: number;
  year: number;
  dataset: Dataset;
  samples: Array<{
    lng: number;
    lat: number;
    positive: boolean;
    embedding: number[];
  }>;
}

const VERSION = 1;
const HEADER_BYTES = 1 + 1 + 2 + 2 + 1 + 4 * 3 + 1;
const DATASETS: Dataset[] = ["aef", "tessera"];
const BANDS: Record<Dataset, number> = { aef: NUM_BANDS, tessera: TESSERA_BANDS };
const sampleBytes = (dataset: Dataset) => 1 + 4 * 2 + BANDS[dataset];
export const SHARE_PREFIX = "#s=";

// Embeddings go back to int8. AEF's signed-square dequantization maps to and
// from its stored values exactly. TESSERA embeddings are unit-length copies
// of stored int8 vectors, so scaling the largest component to 127 recovers
// them to within rounding.
function quantize(dataset: Dataset, embedding: number[]): number[] {
  if (dataset === "aef") {
    return embedding.map((v) =>
      Math.round(Math.sign(v) * Math.sqrt(Math.abs(v)) * 127.5),
    );
  }
  const max = Math.max(...embedding.map(Math.abs)) || 1;
  return embedding.map((v) => Math.round((v / max) * 127));
}

function dequantize(dataset: Dataset, values: number[]): number[] {
  if (dataset === "aef") {
    return values.map((q) => Math.sign(q) * (q / 127.5) ** 2);
  }
  const length = Math.hypot(...values) || 1;
  return values.map((q) => q / length);
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(text: string): Uint8Array {
  const binary = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

export function encodeSearch(search: SharedSearch): string {
  const samples = search.samples.slice(0, 255);
  const buffer = new ArrayBuffer(
    HEADER_BYTES + samples.length * sampleBytes(search.dataset),
  );
  const view = new DataView(buffer);
  let o = 0;
  view.setUint8(o, VERSION), (o += 1);
  view.setUint8(o, DATASETS.indexOf(search.dataset)), (o += 1);
  view.setUint16(o, search.year), (o += 2);
  view.setUint16(o, Math.round(search.threshold * 10000)), (o += 2);
  view.setUint8(o, Math.round(search.counterWeight * 100)), (o += 1);
  view.setFloat32(o, search.center[0]), (o += 4);
  view.setFloat32(o, search.center[1]), (o += 4);
  view.setFloat32(o, search.zoom), (o += 4);
  view.setUint8(o, samples.length), (o += 1);
  for (const s of samples) {
    view.setUint8(o, s.positive ? 1 : 0), (o += 1);
    view.setFloat32(o, s.lng), (o += 4);
    view.setFloat32(o, s.lat), (o += 4);
    for (const q of quantize(search.dataset, s.embedding)) view.setInt8(o++, q);
  }
  return toBase64Url(new Uint8Array(buffer));
}

export function decodeSearch(text: string): SharedSearch | null {
  try {
    const bytes = fromBase64Url(text);
    const view = new DataView(bytes.buffer);
    let o = 0;
    if (view.getUint8(o) !== VERSION) return null;
    o += 1;
    const dataset = DATASETS[view.getUint8(o)];
    if (!dataset) return null;
    o += 1;
    const year = view.getUint16(o);
    o += 2;
    const threshold = view.getUint16(o) / 10000;
    o += 2;
    const counterWeight = view.getUint8(o) / 100;
    o += 1;
    const center: [number, number] = [view.getFloat32(o), view.getFloat32(o + 4)];
    const zoom = view.getFloat32(o + 8);
    o += 12;
    const count = view.getUint8(o);
    o += 1;
    const bands = BANDS[dataset];
    if (bytes.length !== HEADER_BYTES + count * sampleBytes(dataset)) return null;
    const samples: SharedSearch["samples"] = [];
    for (let i = 0; i < count; i++) {
      const positive = view.getUint8(o) === 1;
      const lng = view.getFloat32(o + 1);
      const lat = view.getFloat32(o + 5);
      o += 9;
      const embedding = dequantize(
        dataset,
        Array.from({ length: bands }, (_, b) => view.getInt8(o + b)),
      );
      o += bands;
      samples.push({ lng, lat, positive, embedding });
    }
    return { center, zoom, year, dataset, threshold, counterWeight, samples };
  } catch {
    return null;
  }
}
