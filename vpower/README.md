# Shifts Challenge

This repo contains code regarding the Vessel Power Estimation track of the Shifts Challenge.

### Objective

The Vessel Power Estimation track of Shifts Challenge aims to provide two datasets, a real "in-the-wild" dataset and its
synthetic counterpart, regarding the maritime industry to assessing methods of robustness to distributional shift and
uncertainty estimation. This is a scalar regression task that involves predicting the power consumption of a merchant
vessel for various operational conditions.

The dataset, referred as **_real dataset_**, contains operational data from sensors and a third-party weather provider in
per minute frequency. Furthermore, a second dataset, named **_synthetic dataset_**, is provided. This dataset is
created by combining the real samples with synthetic power labels generated by a generative analytical
physics-based model.

### Dataset Overview

The data are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.
All data are provided as csv files.

Provided datasets are:
* Synthetic dataset (is released in Phase 1 of the challenge): `train.csv`, `dev_in.csv` and `dev_out.csv`
* Real dataset (is released in Phase 2 of the challenge): `train.csv`, `dev_in.csv` and `dev_out.csv`

<br />

* Description of available training features

| Feature name                | Units      | Description                                                                                | Source           |
|:----------------------------|:-----------|:-------------------------------------------------------------------------------------------|:-----------------|
| draft_aft_telegram          | m          | Draft at stern as reported by crew in daily reports                                        | Telegrams        |
| draft_fore_telegram         | m          | Draft at bow as reported by crew in daily reports                                          | Telegrams        |
| stw                         | knots      | Speed through water (i.e. relative to any currents) of the vessel as measured by speed log | Onboard sensor   |
| diff_speed_overground       | knots/3min | Acceleration of the vessel relative to ground                                              | GPS              |
| awind_vcomp_provider        | knots      | Apparent wind speed component relative to the vessel along its direction of motion         | Weather provider |
| awind_ucomp_provider        | knots      | Apparent wind speed component relative to vessel perpendicular to its direction            | Weather provider |
| rcurrent_vcomp              | knots      | Component of currents relative to the vessel along its direction of motion                 | Weather provider |
| rcurrent_ucomp              | knots      | Component of currents relative to vessel perpendicular to its direction                    | Weather provider |
| comb_wind_swell_wave_height | m          | Combined wave height due to wind and sea swell                                             | Weather provider |
| timeSinceDryDock            | minutes    | Time since the last dry dock cleaning of the vessel                                        | Calculated       |
| time_id                     | -          | Run number representing time. It is to be used as index of the records.                    | Calculated       |

<br />

* Description of target features

| Feature name | Units                                                                          | Description                                                                                                                               |
|:-------------|:-------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| power        | kW                                                                             | Propeller shaft power. For synthetic data it is generated by the synthetic model and for real data is measured by an onboard torquemeter. |

### What is included in this repository?

1. Source code that is used by the tutorials.

2. Tutorials:
    1. A notebook to download the `data` and the `trained baseline models` in the current directory.
        * get_files.ipynb
    2. Data analysis tutorial that provides insights about the quality of the data.
        * data_analysis.ipynb
    3. Model training tutorial. This tutorial steps through the data preprocessing and model training process for an
       ensemble of 10 probabilistic MC dropout neural networks
        * ens_prob_mc_dropout_training.ipynb
    4. Tutorial for model evaluation
        * model_evaluation.ipynb

### Who do I talk to?

a.nikitakis@deepsea.ai  
a.athanasopoulos@deepsea.ai  
e.tsompopoulou@deepsea.ai  