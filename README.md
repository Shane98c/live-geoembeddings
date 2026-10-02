# Live Geoembeddings

Click examples on a map and similar spots light up, using satellite
embeddings read directly from cloud storage. Everything is computed in your
browser; there is no server.

Live site: https://shane98c.github.io/live-geoembeddings/

## Modes

Clicks add a match (green) or, with the switch set to "Not a match" or
Shift held, a spot that is not a match (red). Cmd/Ctrl+Z undoes the last
change. The year picker (2017-2025) switches both
the embeddings and the Sentinel-2 imagery; aerial imagery is also available
but is not matched to the year.

- **Find similar spots** (default). Scores each pixel by its similarity to a
  search vector: the average of your matches, minus the average of the spots
  that are not a match times the not-a-match strength (default 0.25; a
  gentler form of [geovibes](https://github.com/cr458/geovibes)' 2 x mean -
  mean). Scores are scaled so your matches average 1, and pixels above the
  match threshold are highlighted.
  **Copy link** encodes the clicks, their embeddings, the threshold and the
  view in the URL.
- **Embeddings as color.** Three embedding dimensions as RGB.

## Embeddings

- **AlphaEarth Foundations** (Google): 64 dimensions per 10 m pixel, loads
  at zoom 12 and closer.
- **TESSERA** (University of Cambridge): 128 dimensions per 10 m pixel,
  stored per UTM zone, loads at zoom 13 and closer. Similarity is the cosine
  of the raw int8 values, so the per-pixel scales are not needed.

## How it works

- **Loading.** Embeddings are read from Zarr v3 stores on Source Cooperative
  with [zarr-layer](https://github.com/carbonplan/zarr-layer), a MapLibre
  custom layer.
- **Rendering.** A fragment shader scores every pixel on screen against the
  search vector. Clicks only change shader uniforms, so no data is refetched.

## Data

| Data | Source | License |
| --- | --- | --- |
| AlphaEarth embeddings, 2017-2025 | [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC-BY 4.0, produced by Google and Google DeepMind |
| TESSERA embeddings, 2017-2025 | [tessera/tessera](https://source.coop/tessera/tessera) | University of Cambridge |
| Sentinel-2 imagery, yearly | [EOX Sentinel-2 cloudless](https://s2maps.eu) (no 2017 mosaic; 2018 is shown) | CC BY-NC-SA 4.0, contains modified Copernicus Sentinel data |
| Aerial imagery | [USGS The National Map](https://basemap.nationalmap.gov/) | Public domain |
| Roads and place names | [OpenFreeMap](https://openfreemap.org) | © OpenMapTiles, OpenStreetMap (ODbL) |

## Development

```sh
npm install
npm run dev
```

Pushes to `main` deploy to GitHub Pages.
