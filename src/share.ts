import { NUM_BANDS } from "./embeddings";
import { TESSERA_BANDS } from "./tessera";

export type Dataset = "aef" | "tessera";
export type SharedMode = "similarity" | "classify";

export interface SharedSample {
  lng: number;
  lat: number;
  positive: boolean;
  embedding: number[];
}

export interface SharedClass {
  name: string;
  color: string;
  samples: SharedSample[];
}

/** A similarity search or a classification small enough to travel in a link. */
export interface SharedSearch {
  mode: SharedMode;
  center: [number, number];
  zoom: number;
  threshold: number;
  counterWeight: number;
  year: number;
  dataset: Dataset;
  samples: SharedSample[];
  classes: SharedClass[];
}

// Version 3 marks the switch to TESSERA v1.1, whose embeddings do not match
// the v1 ones that earlier TESSERA links carry.
const VERSION = 3;
const DATASETS: Dataset[] = ["aef", "tessera"];
const MODES: SharedMode[] = ["similarity", "classify"];
const BANDS: Record<Dataset, number> = { aef: NUM_BANDS, tessera: TESSERA_BANDS };
const MAX_NAME_BYTES = 255;
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

class Writer {
  private bytes: number[] = [];
  u8(v: number) {
    this.bytes.push(v & 0xff);
  }
  i8(v: number) {
    this.u8(v < 0 ? v + 256 : v);
  }
  u16(v: number) {
    this.u8(v >> 8), this.u8(v);
  }
  f32(v: number) {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, v);
    for (let i = 0; i < 4; i++) this.u8(view.getUint8(i));
  }
  text(s: string) {
    const encoded = new TextEncoder().encode(s).slice(0, MAX_NAME_BYTES);
    this.u8(encoded.length);
    for (const b of encoded) this.u8(b);
  }
  done() {
    return Uint8Array.from(this.bytes);
  }
}

class Reader {
  private view: DataView;
  private o = 0;
  constructor(private bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  u8() {
    return this.view.getUint8(this.o++);
  }
  i8() {
    return this.view.getInt8(this.o++);
  }
  u16() {
    const v = this.view.getUint16(this.o);
    this.o += 2;
    return v;
  }
  f32() {
    const v = this.view.getFloat32(this.o);
    this.o += 4;
    return v;
  }
  text() {
    const length = this.u8();
    const s = new TextDecoder().decode(this.bytes.subarray(this.o, this.o + length));
    this.o += length;
    return s;
  }
  atEnd() {
    return this.o === this.bytes.length;
  }
}

function writeSamples(w: Writer, dataset: Dataset, samples: SharedSample[]) {
  const kept = samples.slice(0, 255);
  w.u8(kept.length);
  for (const s of kept) {
    w.u8(s.positive ? 1 : 0);
    w.f32(s.lng);
    w.f32(s.lat);
    for (const q of quantize(dataset, s.embedding)) w.i8(q);
  }
}

function readSamples(r: Reader, dataset: Dataset): SharedSample[] {
  const count = r.u8();
  const samples: SharedSample[] = [];
  for (let i = 0; i < count; i++) {
    const positive = r.u8() === 1;
    const lng = r.f32();
    const lat = r.f32();
    const embedding = dequantize(
      dataset,
      Array.from({ length: BANDS[dataset] }, () => r.i8()),
    );
    samples.push({ lng, lat, positive, embedding });
  }
  return samples;
}

export function encodeSearch(search: SharedSearch): string {
  const w = new Writer();
  w.u8(VERSION);
  w.u8(DATASETS.indexOf(search.dataset));
  w.u16(search.year);
  w.u16(Math.round(search.threshold * 10000));
  w.u8(Math.round(search.counterWeight * 100));
  w.f32(search.center[0]);
  w.f32(search.center[1]);
  w.f32(search.zoom);
  w.u8(MODES.indexOf(search.mode));
  writeSamples(w, search.dataset, search.samples);
  const classes = search.classes.slice(0, 255);
  w.u8(classes.length);
  for (const c of classes) {
    w.text(c.name);
    for (const k of [1, 3, 5]) w.u8(parseInt(c.color.slice(k, k + 2), 16));
    writeSamples(w, search.dataset, c.samples);
  }
  return toBase64Url(w.done());
}

export function decodeSearch(text: string): SharedSearch | null {
  try {
    const r = new Reader(fromBase64Url(text));
    const version = r.u8();
    if (version < 1 || version > VERSION) return null;
    const dataset = DATASETS[r.u8()];
    if (!dataset || (dataset === "tessera" && version < 3)) return null;
    const year = r.u16();
    const threshold = r.u16() / 10000;
    const counterWeight = r.u8() / 100;
    const center: [number, number] = [r.f32(), r.f32()];
    const zoom = r.f32();
    // Version 1 links hold a similarity search only.
    const mode = version === 1 ? "similarity" : MODES[r.u8()];
    if (!mode) return null;
    const samples = readSamples(r, dataset);
    const classes: SharedClass[] = [];
    if (version !== 1) {
      const count = r.u8();
      for (let i = 0; i < count; i++) {
        const name = r.text();
        const color =
          "#" + [r.u8(), r.u8(), r.u8()].map((v) => v.toString(16).padStart(2, "0")).join("");
        classes.push({ name, color, samples: readSamples(r, dataset) });
      }
    }
    if (!r.atEnd()) return null;
    return {
      mode,
      center,
      zoom,
      year,
      dataset,
      threshold,
      counterWeight,
      samples,
      classes,
    };
  } catch {
    return null;
  }
}
