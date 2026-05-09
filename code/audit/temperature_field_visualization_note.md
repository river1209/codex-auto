# Temperature Field Visualization Note

Date: 2026-05-09

## Does the dataset provide spatial positions?

The local metadata file `data/FPV_metafile_mm.xlsx` provides relative plotting
locations for floating-PV module temperature sensors, but it does not provide
surveyed geometric coordinates such as module-center x/y positions in metres.

Relevant metadata fields:

- `module_temperature_plot_location`: relative location codes, e.g. `SE`,
  `SW`, `M`, `NW`, `NE` for Altamonte/Oakville/Orlando and `NW`, `N`, `NE`,
  `M`, `S` for Windsor.
- `module_temperature_plot_location_number`: replicate index `1`, `2`, `3`,
  corresponding to the A/B/C panel-temperature channels.
- `temperature_measurement_location`: `FPV` for floating PV module temperature
  sensors.

Therefore, continuous temperature-field plots should be described as relative
interpolated maps based on metadata location codes, not as true physical
geometry maps.

## Generated Script

`code/scripts/plot_temperature_field_maps.py`

The script reads prediction sample CSVs with `true_*` and `pred_*` columns,
infers relative coordinates from sensor names, and plots:

1. Observed relative temperature field.
2. Predicted relative temperature field.
3. Prediction error field.

The A/B/C replicated sensors are slightly offset around each relative location
to make interpolation possible; this offset is only a plotting convention.

## Generated Figures

- `code/figures/temperature_fields/oakville_h1_timexer_maxspread.png`
- `code/figures/temperature_fields/orlando_h1_crossformer_maxspread.png`
- `code/figures/temperature_fields/windsor_h1_autoformer_maxspread.png`

These figures use the test sample with the largest observed P95-P5 spread in
each prediction CSV, so they emphasize thermal non-uniformity.

## Paper Wording

Suggested figure caption wording:

> Relative interpolated module-temperature fields reconstructed from the
> floating-PV temperature sensor layout codes provided in the dataset metadata.
> The coordinates represent relative sensor groups rather than surveyed module
> positions; A/B/C sensors are displayed with small plotting offsets within
> each group.

