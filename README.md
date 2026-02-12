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

## Acknowledgements

We would like to thank **Greg Silsbe** for making the source code for his paper, 
The CAFE model: A net production model for global
ocean phytoplankton, publicly available. 
Access to his [repository] (https://github.com/gsilsbe/CAFE) significantly eased the implementation of our model framework.


## References

Silsbe GM, Behrenfeld MJ, Halsey KH, Milligan AJ, Westberry TK. The CAFE model: A net production model for global ocean phytoplankton. Global Biogeochemical Cycles. 2016 Dec;30(12):1756–77.

Arst H, Nõges T, Nõges P, Paavel B. In situ measurements and model calculations of primary production in turbid waters. Aquat Biol. 2008 Jul 1;3:19–30.

Arst H, Nõges P, Nõges T, Kauer T, Arst GE. Quantification of a Primary Production Model Using Two Versions of the Spectral Distribution of the Phytoplankton Absorption Coefficient. Environ Model Assess. 2012 Aug;17(4):431–40.

Soomets T, Kutser T, Wüest A, Bouffard D. Spatial and temporal changes of primary production in a deep peri-alpine lake. Inland Waters. 2019 Jan 2;9(1):49–60.

Behrenfeld MJ, Falkowski PG. Photosynthetic rates derived from satellite‐based chlorophyll concentration. Limnology & Oceanography. 1997 Jan;42(1):1–20.

Lee Z, Marra JF. The Use of VGPM to Estimate Oceanic Primary Production: A “Tango” Difficult to Dance. J Remote Sens. 2022 Jan;2022:2022/9851013.

Lee ZP, Carder KL, Marra J, Steward RG, Perry MJ. Estimating primary production at depth from remote sensing. Appl Opt. 1996 Jan 20;35(3):463.

Platt T, Gallegos CL, Harrison WG. Photoinhibition of photosynthesis in natural assemblages of marine phytoplankton. Journal of Marine Research. 1980;(38):687–701.

Stæhr PA, Markager S. Parameterization of the chlorophyll a -specific in vivo light absorption coefficient covering estuarine, coastal and oceanic waters. International Journal of Remote Sensing. 2004 Nov;25(22):5117–30.

Bricaud A, Babin M, Morel A, Claustre H. Variability in the chlorophyll‐specific absorption coefficients of natural phytoplankton: Analysis and parameterization. J Geophys Res. 1995 Jul 15;100(C7):13321–32. 

Westberry TK, Siegel DA. Phytoplankton natural fluorescence variability in the Sargasso Sea. Deep Sea Research Part I: Oceanographic Research Papers. 2003 Mar;50(3):417–34.

Morel A, Smith RC. Relation between total quanta and total energy for aquatic photosynthesis1. Limnology & Oceanography. 1974 Jul;19(4):591–600.

Kirk JTO. Light and photosynthesis in aquatic ecosystems. 3rd ed. Cambridge (GB): Cambridge University Press; 2011. 

