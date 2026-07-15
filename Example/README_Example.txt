Example input files for HGS Sediment-Nutrient Module.

Files:
- SNW_V4o_example_5step_highQ.olf.dat: test OLF with five timesteps and increased Flow Rate values.
- SNW_V4o.chan.dat: matching channel file.
- Model_Config_Example.txt: config that points to these example files.

Q adjustment used for the OLF copy: Flow Rate values are set by timestep to 0.025, 0.040, 0.048, 0.055, 0.060 m3/s.
The sequence is designed to avoid immediate resuspension-cap saturation, so SSC_station_mgL varies across the five example timesteps.
Only the first five timesteps from the source OLF are included.
The original source OLF was not modified.
