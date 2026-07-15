HGS Sediment-Nutrient Module: PEST Starter Files
=================================================

Purpose
-------
This folder provides starter files for connecting the HGS Sediment-Nutrient
Module to PEST or PEST++.

Basic workflow
--------------
1. Copy these files into a calibration working folder, or keep them here.
2. Let PEST use Model_Config.tpl to generate ../Model_Config.txt.
3. Let PEST run Run_Model_For_PEST.bat.
4. Let PEST read ../Outputs/PEST_Run/pest_values.txt with Extract_PEST_Values.ins.

Quick package test
------------------
Double-click Run_Example_PEST_Mode.bat to test PEST-mode output without
launching PEST. It uses Model_Config_PEST_Example.txt and should create:

../Outputs/PEST_Run/pest_values.txt

Important model settings for PEST
---------------------------------
Model_Config.tpl forces:
- pest_mode = True
- timestamp_output_dir = False
- plot_sediment_map_mode = False
- plot_exchange_flux_map_mode = False
- out_dir = 'Outputs/PEST_Run'

This gives PEST a stable output file:
../Outputs/PEST_Run/pest_values.txt

Current simulated values
------------------------
The instruction file reads five values:
- ssc_000
- ssc_001
- ssc_002
- ssc_003
- ssc_004

These names correspond to the five example OLF timesteps. For real
calibration, replace the observation values and weights in the PST file
with your measured SSC or nutrient targets.

Parameters exposed in the template
----------------------------------
SC, C_USLE, K_ch, SDR_A, SDR_B, C_BASE, RESUS_K, QCRIT, MUSLE_W

Notes
-----
The included PST file is a starter skeleton. You should review parameter
bounds, transformations, observation values, weights, and regularization
before using it for final calibration.

Some older PEST installations are sensitive to spaces in file paths. If
that happens, copy the package to a short path such as C:\HGS_PEST_Run
before launching PEST.
