import maplibregl from "maplibre-gl";
import { ZarrLayer } from "@carbonplan/zarr-layer";
import { NUM_BANDS, readPoint } from "./embeddings";
import {
  decodeSearch,
  encodeSearch,
  SHARE_PREFIX,
  type Dataset,
} from "./share";
import { COLORMAP, MIN_ALPHA } from "./highlight";
import { readTesseraPoint, TESSERA_BANDS, TesseraLayers } from "./tessera";
import {
  CLASS_COLOR_GLSL,
  classUniforms,
  MAX_CLASSES,
  PALETTE,
  scoreGlsl,
  type ClassQuery,
} from "./classify";
const SOURCE = "https://data.source.coop/tge-labs/aef-mosaic";
const YEAR_ORIGIN = 2017;
const LATEST_YEAR = 2025;
// Full-resolution only (no pyramid). A 256px region is ~2.3 MB to download
// for AlphaEarth and ~7.4 MB for TESSERA, and a laptop-sized zoom 13 view is
// ~55 regions: about 125 MB or 400 MB.
const MIN_ZOOM = 13;
// Band indices for the false-color view, following Google's AEF examples.
const RGB_BANDS = [1, 16, 9];

// Roads and place names from OpenStreetMap, drawn above the data layers.
const NAME = ["coalesce", ["get", "name:en"], ["get", "name"]];
const REFERENCE_LAYERS: maplibregl.LayerSpecification[] = [
  {
    id: "roads",
    type: "line",
    source: "osm",
    "source-layer": "transportation",
    minzoom: 6,
    filter: [
      "in",
      ["get", "class"],
      [
        "literal",
        ["motorway", "trunk", "primary", "secondary", "tertiary", "minor"],
      ],
    ],
    paint: {
      "line-color": "rgba(255, 255, 255, 0.55)",
      "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.4, 12, 1, 16, 3],
    },
  },
  {
    id: "road-names",
    type: "symbol",
    source: "osm",
    "source-layer": "transportation_name",
    minzoom: 13,
    layout: {
      "symbol-placement": "line",
      "text-field": NAME,
      "text-font": ["Noto Sans Regular"],
      "text-size": 11,
    },
    paint: {
      "text-color": "#fff",
      "text-halo-color": "rgba(0, 0, 0, 0.7)",
      "text-halo-width": 1.2,
    },
  },
  {
    id: "place-names",
    type: "symbol",
    source: "osm",
    "source-layer": "place",
    filter: [
      "in",
      ["get", "class"],
      ["literal", ["city", "town", "village", "hamlet"]],
    ],
    layout: {
      "text-field": NAME,
      "text-font": ["Noto Sans Regular"],
      "text-size": ["match", ["get", "class"], "city", 15, "town", 13, 11],
    },
    paint: {
      "text-color": "#fff",
      "text-halo-color": "rgba(0, 0, 0, 0.75)",
      "text-halo-width": 1.4,
    },
  },
] as maplibregl.LayerSpecification[];
const FIRST_REFERENCE_LAYER = REFERENCE_LAYERS[0].id;

const BAND_INDICES = Array.from({ length: NUM_BANDS }, (_, i) => i);
const MODES = { rgb: 0, similarity: 1, classify: 2 } as const;
type Mode = keyof typeof MODES;

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;
// Touch screens get tap wording and a switch in place of Shift-click.
const touch = matchMedia("(hover: none)").matches;

const modeSwitch = $<HTMLFieldSetElement>("mode");
const modeRadios = () =>
  [...modeSwitch.querySelectorAll<HTMLInputElement>("input[name=mode]")];
function setMode(value: Mode) {
  for (const r of modeRadios()) r.checked = r.value === value;
}
const datasetSelect = $<HTMLSelectElement>("dataset");
const hint = $<HTMLDivElement>("hint");
const threshold = $<HTMLInputElement>("threshold");
const thresholdValue = $<HTMLOutputElement>("threshold-value");
const opacity = $<HTMLInputElement>("opacity");
const opacityValue = $<HTMLOutputElement>("opacity-value");
const samplesLabel = $<HTMLSpanElement>("samples");
const clearButton = $<HTMLButtonElement>("clear");
const status = $<HTMLParagraphElement>("status");
const undoButton = $<HTMLButtonElement>("undo");
const copyLinkButton = $<HTMLButtonElement>("copy-link");
const yearSelect = $<HTMLSelectElement>("year");
const imagerySelect = $<HTMLSelectElement>("imagery");
const yearNote = $<HTMLParagraphElement>("year-note");
const aerialNote = $<HTMLParagraphElement>("aerial-note");
for (let y = LATEST_YEAR; y >= YEAR_ORIGIN; y--)
  yearSelect.add(new Option(String(y)));
const counterControls = $<HTMLLabelElement>("counter-controls");
const counterStrength = $<HTMLInputElement>("counter-strength");
const counterValue = $<HTMLOutputElement>("counter-value");
const classesBox = $<HTMLFieldSetElement>("classes");
const classList = $<HTMLDivElement>("class-list");
const addClassButton = $<HTMLButtonElement>("add-class");

// A shared search or classification in the URL takes over the view and mode.
const shared = location.hash.startsWith(SHARE_PREFIX)
  ? decodeSearch(location.hash.slice(SHARE_PREFIX.length))
  : null;
if (shared) {
  setMode(shared.mode);
  datasetSelect.value = shared.dataset;
  yearSelect.value = String(shared.year);
}

// Raw int8 values arrive as floats (fill already NaN). Dequantize with
// (x / 127.5)^2 * sign(x); embeddings are unit length, so a dot product
// against a unit-length query is the cosine similarity.
const dequant = (band: string) => `(${band} / 127.5) * abs(${band} / 127.5)`;

const embeddingTerms = BAND_INDICES.map(
  (i) => `  e[${i}] = ${dequant(`band_${i}`)};`,
).join("\n");

const [r, g, b] = RGB_BANDS.map((i) => `e[${i}]`);

// Similarity and classify modes draw nothing until there is a query, so the
// imagery shows through. Then each pixel is scored against every class
// vector (one, in similarity mode) and the best score must clear the
// threshold: similarity highlights by score, classify paints the class.
const customFrag = `
  uniform float u_mode;
  uniform float u_threshold;
  uniform float u_count;

  if (isnan(band_0)) {
    discard;
  }

  float e[${NUM_BANDS}];
${embeddingTerms}

  if (u_mode < 0.5) {
    vec3 rgb = clamp((vec3(${r}, ${g}, ${b}) + 0.3) / 0.6, 0.0, 1.0);
    fragColor = vec4(rgb * opacity, opacity);
  } else if (u_count < 0.5) {
    discard;
  } else {
${scoreGlsl(NUM_BANDS)}
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

type Sample = {
  embedding: number[];
  positive: boolean;
  marker: maplibregl.Marker;
};
const samples: Sample[] = [];

// Classify mode: each class has its own examples and counter-examples.
type LandClass = {
  id: number;
  name: string;
  color: string;
  samples: Sample[];
};
let classes: LandClass[] = [];
let activeClassId = 0;
let nextClassId = 1;
const activeClass = () => classes.find((c) => c.id === activeClassId);
const allClassSamples = () => classes.flatMap((c) => c.samples);

function newClass(name: string): LandClass {
  const used = new Set(classes.map((c) => c.color));
  const color = PALETTE.find((c) => !used.has(c)) ?? PALETTE[classes.length % PALETTE.length];
  return { id: nextClassId++, name, color, samples: [] };
}

// Undo restores the state from before each change: a click, Start over, a
// change of embeddings, or adding or removing a class.
type Snapshot = {
  samples: Sample[];
  classes: LandClass[];
  activeClassId: number;
  dataset: string;
};
const undoStack: Snapshot[] = [];
const MAX_HISTORY = 50;
// The select already shows the new value when its change event fires.
let previousDataset = datasetSelect.value;

function pushHistory() {
  undoStack.push({
    samples: [...samples],
    classes: classes.map((c) => ({ ...c, samples: [...c.samples] })),
    activeClassId,
    dataset: previousDataset,
  });
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  undoButton.disabled = false;
}

function undo() {
  const previous = undoStack.pop();
  undoButton.disabled = undoStack.length === 0;
  if (!previous) return;
  for (const { marker } of [...samples, ...allClassSamples()]) marker.remove();
  samples.splice(0, samples.length, ...previous.samples);
  classes = previous.classes;
  activeClassId = previous.activeClassId;
  for (const { marker } of samples) {
    marker.addTo(map).getElement().hidden = mode() !== "similarity";
  }
  for (const { marker } of allClassSamples()) {
    marker.addTo(map).getElement().hidden = mode() !== "classify";
  }
  datasetSelect.value = previousDataset = previous.dataset;
  renderClasses();
  applyDataset();
}

const mode = () =>
  (modeRadios().find((r) => r.checked)?.value ?? "similarity") as Mode;
const dataset = () => datasetSelect.value as Dataset;
const minZoom = () => MIN_ZOOM;
const year = () => Number(yearSelect.value);

// EOX publishes a cloudless Sentinel-2 mosaic per year, except 2017.
const mosaicYear = (y: number) => (y === 2017 ? 2018 : y);
const sentinelTiles = (y: number) =>
  `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${mosaicYear(y)}_3857/default/g/{z}/{y}/{x}.jpg`;

const sentinelSource = (y: number): maplibregl.RasterSourceSpecification => ({
  type: "raster",
  tiles: [sentinelTiles(y)],
  tileSize: 256,
  attribution:
    'Sentinel-2 cloudless by <a href="https://s2maps.eu">EOX IT Services GmbH</a> (contains modified Copernicus Sentinel data)',
});

const sentinelLayer = (): maplibregl.RasterLayerSpecification => ({
  id: "sentinel",
  type: "raster",
  source: "sentinel",
  layout: {
    visibility: imagerySelect.value === "sentinel" ? "visible" : "none",
  },
});
if (shared) {
  threshold.value = String(shared.threshold);
  counterStrength.value = String(shared.counterWeight);
}
counterValue.textContent = counterStrength.value;

/** The query vectors to score against: the classes', or the one search. */
function queries(): ClassQuery[] {
  if (mode() === "classify") {
    return classes.flatMap((c) => {
      const vector = queryVector(c.samples);
      return vector ? [{ vector, color: c.color }] : [];
    });
  }
  const vector = queryVector(samples);
  return vector ? [{ vector, color: "#ffffff" }] : [];
}

function layerUniforms(current: Dataset, dims: number) {
  return {
    u_mode: MODES[mode()],
    u_threshold: Number(threshold.value),
    ...classUniforms(dims, dataset() === current ? queries() : []),
  };
}

const uniforms = () => layerUniforms("aef", NUM_BANDS);
const tesseraUniforms = () => layerUniforms("tessera", TESSERA_BANDS);

function mean(embeddings: number[][]): number[] {
  const out = new Array(embeddings[0].length).fill(0);
  for (const e of embeddings)
    e.forEach((v, i) => (out[i] += v / embeddings.length));
  return out;
}

// How much counter-examples subtract from the search vector (user
// adjustable). The low default follows Rocchio relevance feedback, where
// negative examples get a small weight: with a fixed threshold, heavier
// weights (geovibes' 2 x mean - mean is 0.5) also drop true matches when
// examples and counter-examples look alike.
const counterWeight = () => Number(counterStrength.value);

/**
 * The search vector for a set of samples: mean(examples) - counterWeight() x
 * mean(counter-examples), a gentler form of geovibes' query. Subtracting
 * counter-examples lowers every score, examples included, so the vector is
 * scaled for the examples to average a score of 1: the threshold is a
 * fraction of how well the examples themselves match. With one example this
 * is a plain cosine. In classify mode each class gets its own vector, so
 * the scaling also puts the classes' scores on the same footing.
 */
function queryVector(from: Sample[]): number[] | null {
  const pos = from.filter((s) => s.positive).map((s) => s.embedding);
  if (pos.length === 0) return null;
  const neg = from.filter((s) => !s.positive).map((s) => s.embedding);
  let q = mean(pos);
  if (neg.length) {
    const n = mean(neg);
    q = q.map((v, i) => v - counterWeight() * n[i]);
  }
  const examplesScore =
    pos.reduce((sum, e) => sum + e.reduce((d, v, i) => d + v * q[i], 0), 0) /
    pos.length;
  if (examplesScore <= 0) return null;
  return q.map((v) => v / examplesScore);
}

const selector = (yearIdx: number) => ({
  time: { selected: yearIdx, type: "index" as const },
  band: { selected: BAND_INDICES, type: "index" as const },
});

// Each AEF layer reports its own loading state; the indicator shows while
// any of them is fetching.
const loadingLayers = new Set<string>();
// The notice at the top of the map says when to zoom in, shows loading,
// and says when the map is ready for clicks.
const notice = $<HTMLDivElement>("notice");
const noticeText = $<HTMLSpanElement>("notice-text");
const noticeSpinner = $<HTMLSpanElement>("notice-spinner");
const READY_MS = 3000;
let readyUntil = 0;
// Set when the view zooms in far enough, so "Ready" follows that load only.
let awaitingReady = true;

function updateNotice() {
  let text = "";
  if (map.getZoom() < minZoom()) text = "Zoom in to start";
  else if (loadingLayers.size > 0) text = "Loading embeddings…";
  else if (Date.now() < readyUntil)
    text = touch ? "Ready: tap a spot" : "Ready: click a spot";
  noticeText.textContent = text;
  noticeSpinner.hidden = loadingLayers.size === 0 || map.getZoom() < minZoom();
  notice.hidden = !text;
}

function setLayerLoading(id: string, loading: boolean) {
  if (loading) loadingLayers.add(id);
  else loadingLayers.delete(id);
  // Wait for loading to stay quiet briefly, so a late chunk does not follow
  // "Ready" with another "Loading".
  if (loadingLayers.size === 0) {
    setTimeout(() => {
      if (loadingLayers.size > 0 || !awaitingReady || map.getZoom() < minZoom())
        return;
      awaitingReady = false;
      readyUntil = Date.now() + READY_MS;
      updateNotice();
      setTimeout(updateNotice, READY_MS + 50);
    }, 400);
  }
  updateNotice();
}

const aefStyle = (id: string) => ({
  clim: [0, 1] as [number, number],
  colormap: COLORMAP,
  opacity: Number(opacity.value),
  customFrag,
  uniforms: uniforms(),
  onLoadingStateChange: ({
    loading,
    error,
  }: {
    loading: boolean;
    error?: Error | null;
  }) => {
    setLayerLoading(id, loading && !error);
    if (error) status.textContent = `Error: ${error.message}`;
  },
});

const layer = new ZarrLayer({
  id: "aef",
  source: SOURCE,
  variable: "embeddings",
  selector: selector(year() - YEAR_ORIGIN),
  minzoom: MIN_ZOOM,
  ...aefStyle("aef"),
});

const map: maplibregl.Map = new maplibregl.Map({
  container: "map",
  center: shared?.center ?? [-121.75, 45.33],
  zoom: shared?.zoom ?? 14,
  maxZoom: 18,
  // Shift-click adds samples; box zoom would swallow it.
  boxZoom: false,
  attributionControl: false,
  style: {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      osm: {
        type: "vector",
        url: "https://tiles.openfreemap.org/planet",
        attribution:
          '<a href="https://openfreemap.org">OpenFreeMap</a> © <a href="https://www.openmaptiles.org/">OpenMapTiles</a> Data from <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      },
      imagery: {
        type: "raster",
        tiles: [
          "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        maxzoom: 16,
        attribution: "Imagery: USGS The National Map",
      },
      sentinel: sentinelSource(year()),
    },
    layers: [
      {
        id: "imagery",
        type: "raster",
        source: "imagery",
        layout: {
          visibility: imagerySelect.value === "aerial" ? "visible" : "none",
        },
      },
      sentinelLayer(),
      ...REFERENCE_LAYERS,
    ],
  },
});
// Top right stays clear of the bottom sheet on phones (style.css moves it to
// the bottom on wider screens). On narrow maps
// MapLibre shrinks the attribution to an info button and opens it as it
// does; here it closes then, since open it covers much of a phone screen.
// Leaving `compact` unset lets it follow the map's width.
map.addControl(
  new maplibregl.AttributionControl({
    customAttribution: '<a href="https://maplibre.org/">MapLibre</a>',
  }),
  "top-right",
);
let attributionCompact = false;
const closeAttributionWhenCompacted = () => {
  const control = map.getContainer().querySelector(".maplibregl-ctrl-attrib");
  const compact = !!control?.classList.contains("maplibregl-compact");
  if (compact && !attributionCompact) {
    control!.classList.remove("maplibregl-compact-show");
  }
  attributionCompact = compact;
};
map.on("load", closeAttributionWhenCompacted);
map.on("resize", closeAttributionWhenCompacted);

// The sheet starts collapsed on phones.
const narrow = matchMedia("(max-width: 640px)").matches;
const panel = $<HTMLElement>("panel");
const panelToggle = $<HTMLButtonElement>("panel-toggle");
function setPanelCollapsed(collapsed: boolean) {
  panel.classList.toggle("collapsed", collapsed);
  panelToggle.setAttribute("aria-expanded", String(!collapsed));
  panelToggle.textContent = collapsed ? "Settings" : "Hide settings";
}
setPanelCollapsed(narrow);
panelToggle.addEventListener("click", () =>
  setPanelCollapsed(!panel.classList.contains("collapsed")),
);

const tessera = new TesseraLayers(
  map,
  (id) => ({
    ...aefStyle(id),
    minzoom: MIN_ZOOM,
    uniforms: tesseraUniforms(),
  }),
  FIRST_REFERENCE_LAYER,
  MIN_ZOOM,
  year(),
);

/** Show the chosen dataset's layers and hide the other's. */
function applyDataset() {
  const current = dataset();
  if (!map.getLayer("aef")) return;
  map.setLayoutProperty(
    "aef",
    "visibility",
    current === "aef" ? "visible" : "none",
  );
  tessera.update(current === "tessera");
  // Hidden layers stop fetching and may never report finishing.
  const hidden = current === "aef" ? tessera.ids() : ["aef"];
  for (const id of hidden) loadingLayers.delete(id);
  refreshUniforms();
  updatePanel();
  updateNotice();
  updateCursor();
}

let shownYear = year();

/** Point the embeddings and the Sentinel-2 imagery at the current year. */
function applyYear() {
  const y = year();
  if (y !== shownYear) {
    shownYear = y;
    layer.setSelector(selector(y - YEAR_ORIGIN));
    tessera.setYear(y);
    // Replace the imagery rather than retiling it: retiled sources keep the
    // old year's tiles on screen until each new one arrives.
    map.removeLayer("sentinel");
    map.removeSource("sentinel");
    map.addSource("sentinel", sentinelSource(y));
    map.addLayer(sentinelLayer(), "imagery");
  }
  yearNote.textContent =
    imagerySelect.value === "sentinel" && mosaicYear(y) !== y
      ? `There is no Sentinel-2 mosaic for ${y}; the imagery is from ${mosaicYear(y)}.`
      : "";
  yearNote.hidden = !yearNote.textContent;
  aerialNote.textContent = `The aerial photos are from various years, so they may not match the ${y} embeddings.`;
  aerialNote.hidden = imagerySelect.value !== "aerial";
}

yearSelect.addEventListener("change", applyYear);
imagerySelect.addEventListener("change", () => {
  const sentinel = imagerySelect.value === "sentinel";
  map.setLayoutProperty(
    "sentinel",
    "visibility",
    sentinel ? "visible" : "none",
  );
  map.setLayoutProperty("imagery", "visibility", sentinel ? "none" : "visible");
  applyYear();
});
applyYear();

const referenceVisible = $<HTMLInputElement>("reference-visible");
referenceVisible.addEventListener("change", () => {
  for (const { id } of REFERENCE_LAYERS) {
    map.setLayoutProperty(
      id,
      "visibility",
      referenceVisible.checked ? "visible" : "none",
    );
  }
});

function refreshUniforms() {
  layer.setUniforms(uniforms());
  tessera.setUniforms(tesseraUniforms());
}

function addMarker(
  lngLat: maplibregl.LngLat,
  kind: "positive" | "negative",
  color?: string,
) {
  const el = document.createElement("div");
  el.className = `sample-marker ${kind}`;
  if (color) el.style.setProperty("--marker-color", color);
  return new maplibregl.Marker({ element: el }).setLngLat(lngLat).addTo(map);
}

// Whether a click adds a match or not a match. Shift-click adds the other.
const addMode = $<HTMLFieldSetElement>("add-mode");
$<HTMLElement>("add-mode-label").textContent = touch
  ? "Taps add"
  : "Clicks add";
const addsMatch = () =>
  addMode.querySelector<HTMLInputElement>("input:checked")?.value ===
  "positive";
const positiveLabel = $<HTMLSpanElement>("positive-label");
const negativeLabel = $<HTMLSpanElement>("negative-label");

const tapOrClick = touch ? "Tap" : "Click";
const HINTS: Record<Mode, { intro: string; steps: string[] }> = {
  similarity: {
    intro: `${tapOrClick} a spot to highlight places like it. Everything runs in the browser.`,
    steps: [
      "Add more matches to sharpen what it looks for.",
      "Add spots that are not a match to steer it away from them.",
    ],
  },
  rgb: {
    intro:
      "Three of the embedding dimensions shown as red, green and blue. Similar colors mean similar embeddings.",
    steps: [],
  },
  classify: {
    intro: `Pick a class, then ${tapOrClick.toLowerCase()} spots that are examples of it. Each spot on the map takes the color of the class it matches best.`,
    steps: [
      "Give every class a few examples, and some spots that are not that class.",
      "Rename the classes, change their colors, or add your own.",
      "Spots that match no class well enough stay clear; lower the threshold to fill them in.",
    ],
  },
};

function renderHint(current: Mode) {
  const { intro, steps } = HINTS[current];
  const p = document.createElement("p");
  p.textContent = intro;
  const ol = document.createElement("ol");
  for (const step of steps) {
    const li = document.createElement("li");
    li.textContent = step;
    ol.append(li);
  }
  hint.replaceChildren(p, ...(steps.length ? [ol] : []));
}

function updatePanel() {
  const current = mode();
  const classify = current === "classify";
  copyLinkButton.hidden = current === "rgb";
  counterControls.hidden = current === "rgb";
  colorbar.hidden = current !== "similarity";
  addMode.hidden = current === "rgb";
  classesBox.hidden = !classify;
  samplesLabel.hidden = classify;
  positiveLabel.textContent = classify ? "Example" : "Match";
  negativeLabel.textContent = classify ? "Not this" : "Not a match";
  renderHint(current);
  const pos = samples.filter((s) => s.positive).length;
  const neg = samples.length - pos;
  samplesLabel.textContent = `${pos} match${pos === 1 ? "" : "es"} · ${neg} not a match`;
  copyLinkButton.disabled = classify
    ? !classes.some((c) => c.samples.some((s) => s.positive))
    : pos === 0;
  addClassButton.disabled = classes.length >= MAX_CLASSES;
  for (const c of classes) {
    const count = classList.querySelector(`[data-count="${c.id}"]`);
    if (!count) continue;
    const p = c.samples.filter((s) => s.positive).length;
    count.textContent = `${p} · ${c.samples.length - p}`;
  }
}

/** Rebuild the class rows: a swatch, an editable name, and sample counts. */
function renderClasses() {
  classList.replaceChildren(
    ...classes.map((c) => {
      const row = document.createElement("div");
      row.className = "class-row";
      row.classList.toggle("active", c.id === activeClassId);

      const pick = document.createElement("input");
      pick.type = "radio";
      pick.name = "class";
      pick.checked = c.id === activeClassId;
      pick.title = "Clicks add to this class";
      pick.addEventListener("change", () => {
        activeClassId = c.id;
        for (const r of classList.children) {
          r.classList.toggle("active", r === row);
        }
      });

      const color = document.createElement("input");
      color.type = "color";
      color.value = c.color;
      color.title = "Class color";
      color.addEventListener("input", () => {
        c.color = color.value;
        for (const { marker } of c.samples) {
          marker.getElement().style.setProperty("--marker-color", c.color);
        }
        refreshUniforms();
      });

      const name = document.createElement("input");
      name.type = "text";
      name.value = c.name;
      name.placeholder = "Class name";
      name.setAttribute("aria-label", "Class name");
      name.addEventListener("input", () => (c.name = name.value));
      name.addEventListener("focus", () => {
        if (!pick.checked) pick.click();
      });

      const count = document.createElement("span");
      count.className = "class-count";
      count.dataset.count = String(c.id);
      count.title = "Examples · not this class";

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "class-remove";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${c.name || "class"}`);
      remove.addEventListener("click", () => {
        pushHistory();
        for (const { marker } of c.samples) marker.remove();
        classes = classes.filter((other) => other !== c);
        if (activeClassId === c.id) activeClassId = classes[0]?.id ?? 0;
        renderClasses();
        refreshUniforms();
        updatePanel();
      });

      row.append(pick, color, name, count, remove);
      return row;
    }),
  );
  updatePanel();
}

addClassButton.title = `Up to ${MAX_CLASSES} class${MAX_CLASSES === 1 ? "" : "es"} on this device`;
addClassButton.addEventListener("click", () => {
  if (classes.length >= MAX_CLASSES) return;
  pushHistory();
  const added = newClass("");
  classes.push(added);
  activeClassId = added.id;
  renderClasses();
  classList
    .querySelector<HTMLInputElement>(".class-row.active input[type=text]")
    ?.focus();
});

if (shared?.classes.length) {
  for (const { name, color, samples: shown } of shared.classes.slice(0, MAX_CLASSES)) {
    const c = newClass(name);
    c.color = color;
    c.samples = shown.map(({ lng, lat, positive, embedding }) => ({
      embedding,
      positive,
      marker: addMarker(
        new maplibregl.LngLat(lng, lat),
        positive ? "positive" : "negative",
        color,
      ),
    }));
    classes.push(c);
  }
} else {
  for (const name of ["Forest", "Water", "Roads"].slice(0, MAX_CLASSES))
    classes.push(newClass(name));
}
activeClassId = classes[0]?.id ?? 0;

async function onSampleClick(e: maplibregl.MapMouseEvent) {
  const positive = addsMatch() !== e.originalEvent.shiftKey;
  const target = mode() === "classify" ? activeClass() : null;
  if (mode() === "classify" && !target) {
    status.textContent = "Add a class first";
    return;
  }
  const { lng, lat } = e.lngLat;
  const clicked = dataset();
  const zoneLayer = tessera.layerAt(lng);
  const embedding =
    clicked === "aef"
      ? await readPoint(layer, lng, lat)
      : zoneLayer
        ? await readTesseraPoint(zoneLayer, lng, lat)
        : null;
  if (clicked !== dataset()) return;
  if (!embedding) {
    status.textContent = "No embedding at that point";
    return;
  }
  pushHistory();
  (target?.samples ?? samples).push({
    embedding,
    positive,
    marker: addMarker(e.lngLat, positive ? "positive" : "negative", target?.color),
  });
  refreshUniforms();
  updatePanel();
}

map.on("click", async (e) => {
  if (map.getZoom() < minZoom() || mode() === "rgb") return;
  status.textContent = "Reading embedding…";
  try {
    await onSampleClick(e);
    if (status.textContent === "Reading embedding…") {
      status.textContent = "";
    }
  } catch (err) {
    status.textContent = `Query failed: ${(err as Error).message}`;
    console.error(err);
  }
});

function clearSamples() {
  for (const { marker } of samples) marker.remove();
  samples.length = 0;
}

/** Drop the classes' samples; the classes themselves stay. */
function clearClassSamples() {
  for (const c of classes) {
    for (const { marker } of c.samples) marker.remove();
    c.samples = [];
  }
}

clearButton.addEventListener("click", () => {
  pushHistory();
  if (mode() === "classify") clearClassSamples();
  else clearSamples();
  addMode.querySelector<HTMLInputElement>("input[value=positive]")!.checked = true;
  refreshUniforms();
  updatePanel();
});

undoButton.addEventListener("click", undo);
document.addEventListener("keydown", (e) => {
  const editing = (e.target as HTMLElement).closest("input, textarea, select");
  if (
    (e.metaKey || e.ctrlKey) &&
    !e.shiftKey &&
    e.key.toLowerCase() === "z" &&
    !editing
  ) {
    e.preventDefault();
    undo();
  }
});
modeSwitch.addEventListener("change", () => {
  for (const s of samples)
    s.marker.getElement().hidden = mode() !== "similarity";
  for (const s of allClassSamples())
    s.marker.getElement().hidden = mode() !== "classify";
  refreshUniforms();
  updatePanel();
  updateCursor();
});

const colorbar = $<HTMLDivElement>("colorbar");
const colorbarLow = $<HTMLSpanElement>("colorbar-low");

/** The colormap with the shader's opacity fade, as a CSS gradient. */
function colorbarRamp(): string {
  const stops = COLORMAP.map((hex, i) => {
    const t = i / (COLORMAP.length - 1);
    const alpha = Number(opacity.value) * (MIN_ALPHA + (1 - MIN_ALPHA) * t);
    const [r, g, b] = [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)}) ${(t * 100).toFixed(0)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function updateColorbar() {
  colorbar.style.setProperty("--ramp", colorbarRamp());
  colorbarLow.textContent = `Threshold ${Number(threshold.value).toFixed(2)}`;
}

threshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(threshold.value).toFixed(3);
  updateColorbar();
  refreshUniforms();
});
thresholdValue.textContent = Number(threshold.value).toFixed(3);
updateColorbar();

opacity.addEventListener("input", () => {
  opacityValue.textContent = opacity.value;
  updateColorbar();
  layer.setOpacity(Number(opacity.value));
  tessera.setOpacity(Number(opacity.value));
});
opacityValue.textContent = opacity.value;

/** A pointer where clicks add examples, the default map cursor elsewhere. */
function updateCursor() {
  const clickable = map.getZoom() >= minZoom() && mode() !== "rgb";
  map.getCanvas().style.cursor = clickable ? "pointer" : "";
}

map.on("zoom", () => {
  if (map.getZoom() < minZoom()) awaitingReady = true;
  updateNotice();
  updateCursor();
});
map.on("load", updateCursor);
map.on("load", updateNotice);
map.on("moveend", () => {
  if (dataset() === "tessera") tessera.update(true);
});

datasetSelect.addEventListener("change", () => {
  pushHistory();
  previousDataset = datasetSelect.value;
  clearSamples();
  clearClassSamples();
  applyDataset();
});

counterStrength.addEventListener("input", () => {
  counterValue.textContent = counterStrength.value;
  refreshUniforms();
});
map.on("load", () => {
  map.addLayer(layer, FIRST_REFERENCE_LAYER);
  applyDataset();
});

if (shared) {
  for (const { lng, lat, positive, embedding } of shared.samples) {
    samples.push({
      embedding,
      positive,
      marker: addMarker(
        new maplibregl.LngLat(lng, lat),
        positive ? "positive" : "negative",
      ),
    });
  }
}
for (const s of samples) s.marker.getElement().hidden = mode() !== "similarity";
for (const s of allClassSamples())
  s.marker.getElement().hidden = mode() !== "classify";
refreshUniforms();
renderClasses();

copyLinkButton.addEventListener("click", async () => {
  const center = map.getCenter();
  const toShared = (from: Sample[]) =>
    from.map(({ embedding, positive, marker }) => {
      const { lng, lat } = marker.getLngLat();
      return { lng, lat, positive, embedding };
    });
  const classify = mode() === "classify";
  const encoded = encodeSearch({
    mode: classify ? "classify" : "similarity",
    center: [center.lng, center.lat],
    zoom: map.getZoom(),
    threshold: Number(threshold.value),
    counterWeight: counterWeight(),
    year: year(),
    dataset: dataset(),
    samples: classify ? [] : toShared(samples),
    classes: classify
      ? classes.map((c) => ({
          name: c.name,
          color: c.color,
          samples: toShared(c.samples),
        }))
      : [],
  });
  const url = `${location.origin}${location.pathname}${SHARE_PREFIX}${encoded}`;
  window.history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    status.textContent = `Link copied. Anyone who opens it sees this ${classify ? "classification" : "search"}.`;
  } catch {
    status.textContent = "The link is in the address bar; copy it from there.";
  }
});
Object.assign(window, { map, layer, tessera });
