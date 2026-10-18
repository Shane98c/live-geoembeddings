# Live Geoembeddings

Click examples on a map and similar spots light up, or define a few land
cover classes and get a classification map, using satellite embeddings read
directly from cloud storage. Everything is computed in your browser; there
is no server.

Live site: https://shane98c.github.io/live-geoembeddings/

## Modes

Clicks add a match (a circle) or, with the switch set to "Not a match" or
Shift held, a spot that is not a match (an X). Cmd/Ctrl+Z undoes the last
change. The year picker (2017-2025) switches both
the embeddings and the Sentinel-2 imagery; aerial imagery is also available
but is not matched to the year.

- **Similar** (default). Scores each pixel by its similarity to a
  search vector: the average of your matches, minus the average of the spots
  that are not a match times the not-a-match strength (default 0.25; a
  gentler form of [geovibes](https://github.com/cr458/geovibes)' 2 x mean -
  mean). Scores are scaled so your matches average 1, and pixels above the
  match threshold are highlighted.
- **Classify.** Define a few classes (Forest, Water and Roads to start;
  rename them, recolor them, add your own) and click examples of each, plus
  spots that are not that class. Each class gets a search vector built as
  above, and every pixel takes the color of the class that scores it
  highest. Pixels that no class scores above the match threshold stay clear.
  The limit is six classes on most devices (see Rendering below).
- **Colors.** Three embedding dimensions as RGB, or any one dimension on a
  colormap (greyscale, viridis, magma or blue to red) with a key. The
  colormap spans -0.3 to 0.3 for AlphaEarth and -0.2 to 0.2 for TESSERA
  (TESSERA vectors are scaled to unit length first), roughly the 1st to
  99th percentile of values.

**Copy link** encodes the clicks, their embeddings, the classes, the
threshold and the view in the URL. TESSERA links made before the switch to
v1.1 carry v1 embeddings, which do not match v1.1, so they open the default
view instead.

## Embeddings

Both load at zoom 13 and closer; a laptop-sized view is roughly 125 MB of
AlphaEarth or 400 MB of TESSERA.

- **AlphaEarth Foundations** (Google): 64 dimensions per 10 m pixel.
- **TESSERA v1.1** (University of Cambridge): 128 dimensions per 10 m pixel,
  from dClimate's wall-to-wall global run (`zarr/v1.1-dclimate`), stored per
  UTM zone. Similarity is the cosine of the raw int8 values, so the
  per-pixel scales are not needed.

## How it works

- **Loading.** Embeddings are read from Zarr v3 stores on Source Cooperative
  with [zarr-layer](https://github.com/carbonplan/zarr-layer), a MapLibre
  custom layer. TESSERA is read straight from the S3 bucket behind Source
  Cooperative, as [GeoTessera](https://github.com/ucam-eo/geotessera) does,
  since the `data.source.coop` gateway can throttle that volume of requests.
- **Rendering.** A fragment shader scores every pixel on screen against the
  search vector, or against every class's vector. Clicks only change shader
  uniforms, so no data is refetched. The class vectors share one uniform
  array sized at compile time, and each slot costs fragment uniform vectors
  whether used or not, so the class limit is read from the device's budget
  at startup: six where WebGL2 reports the usual 1024 vectors, fewer where
  it reports less. A data texture would lift the limit, but zarr-layer's
  custom shaders cannot bind one yet.

## Data

| Data | Source | License |
| --- | --- | --- |
| AlphaEarth embeddings, 2017-2025 | [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC-BY 4.0, produced by Google and Google DeepMind |
| TESSERA v1.1 embeddings, 2017-2025 | [tessera/tessera](https://source.coop/tessera/tessera), `zarr/v1.1-dclimate` | University of Cambridge; global run by dClimate |
| Sentinel-2 imagery, yearly | [EOX Sentinel-2 cloudless](https://s2maps.eu) (no 2017 mosaic; 2018 is shown) | CC BY-NC-SA 4.0, contains modified Copernicus Sentinel data |
| Aerial imagery | [USGS The National Map](https://basemap.nationalmap.gov/) | Public domain |
| Roads and place names | [OpenFreeMap](https://openfreemap.org) | © OpenMapTiles, OpenStreetMap (ODbL) |

## Development

```sh
npm install
npm run dev
```

Pushes to `main` deploy to GitHub Pages.
