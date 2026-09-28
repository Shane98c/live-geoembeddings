import type { ZarrLayer } from "@carbonplan/zarr-layer";

export const NUM_BANDS = 64;

const dequantize = (v: number) => Math.sign(v) * (v / 127.5) ** 2;

type BandSeries = Record<string, number[]>;

/**
 * Split a multi-band query result into per-pixel embeddings. Results nest by
 * band label, one array per band with one entry per pixel; pixels missing any
 * band are dropped.
 */
function toEmbeddings(result: unknown): number[][] {
  const byBand = (result as Record<string, unknown>)["embeddings"];
  if (!byBand || typeof byBand !== "object") return [];
  const series = Object.values(byBand as BandSeries);
  if (series.length !== NUM_BANDS) return [];
  const embeddings: number[][] = [];
  for (let p = 0; p < series[0].length; p++) {
    const raw = series.map((values) => values[p]);
    if (raw.every(Number.isFinite)) embeddings.push(raw.map(dequantize));
  }
  return embeddings;
}

export async function readPoint(
  layer: ZarrLayer,
  lng: number,
  lat: number,
): Promise<number[] | null> {
  const result = await layer.queryData(
    { type: "Point", coordinates: [lng, lat] },
    undefined,
    { includeSpatialCoordinates: false },
  );
  return toEmbeddings(result)[0] ?? null;
}
