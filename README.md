# Assessment of primary production in optically complex waters: Towards a generalized bio-optical modelling approach

Jonas Wydler, 2026_02_11

Code for comparing depth-resolved primary production (PP) models against in-situ measurements from European lakes and the Baltic Sea.

## Data Sources

- Lake Geneva (SHL2 station)
- Baltic Sea (PRICKEN, SLAGGO, BROA E)
- Estonian lakes (Peipsi, Harku, Vortsjarv)

Irradiance data from ERA5 reanalysis.

## Models

Each model configuration is identified by a 3-letter code: `{vertical_resolution}{upwelling_correction}{PI_function}`

**Vertical resolution:**
- A = uniform chlorophyll-a profile
- B = depth-resolved chlorophyll-a profile

**Upwelling correction:**
- A = constant factor (1.4)
- B = depth-dependent (Arst et al., 2012)

**P-I function:**
- A = Lee-like
- B = Arst-like
- C = CAFE-like
- D = VGPM

## Usage

```
python main_p1.py
```

Outputs are saved to `results/tables/` and `results/figures/`.

## Requirements

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- geopandas
- netCDF4
- cdsapi (for ERA5 data download)
