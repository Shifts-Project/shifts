![img](https://github.com/yandex-research/uncertainty-challenge/blob/main/Logoshifts_full_white.png)

# Shifts Challenge

This repository contains data readers and examples for the different datasets provided by the [Shifts Project](https://shifts.ai). 

The Shifts Dataset contains curated and labeled examples of real, 'in-the-wild' distributional shifts across three large-scale tasks. Specifically, it contains white matter multiple sclerosis lesions segmentation and vessel power estimation tasks' data currently used in [Shifts Challenge 2022](https://shifts.grand-challenge.org/), as well as tabular weather prediction, machine translation, and vehicle motion prediction tasks' data used in Shifts Challenge 2021. Dataset shift is ubiquitous in all of these tasks and modalities. 

The dataset, assessment metrics and benchmark results are detailed in our associated papers:

* [Shifts 2.0: Extending The Dataset of Real Distributional Shifts](https://arxiv.org/pdf/2206.15407) (2022)
* [Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks](https://arxiv.org/pdf/2107.07455.pdf) (2021)


If you use Shifts datasets in your work, please cite our papers using the following Bibtex:
```
@misc{https://doi.org/10.48550/arxiv.2206.15407,
  author = {Malinin, Andrey and Athanasopoulos, Andreas and Barakovic, Muhamed and Cuadra, Meritxell Bach and Gales, Mark J. F. and Granziera, Cristina and Graziani, Mara and Kartashev, Nikolay and Kyriakopoulos, Konstantinos and Lu, Po-Jui and Molchanova, Nataliia and Nikitakis, Antonis and Raina, Vatsal and La Rosa, Francesco and Sivena, Eli and Tsarsitalidis, Vasileios and Tsompopoulou, Efi and Volf, Elena},
  title = {Shifts 2.0: Extending The Dataset of Real Distributional Shifts},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2206.15407}
  url = {https://arxiv.org/abs/2206.15407}
}
```
```
@article{shifts2021,
  author    = {Malinin, Andrey and Band, Neil and Ganshin, Alexander, and Chesnokov, German and Gal, Yarin, and Gales, Mark J. F. and Noskov, Alexey and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and Raina, Vyas and Roginskiy, Denis and Shmatova, Mariya and Tigar, Panos and Yangel, Boris},
  title     = {Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks},
  journal   =  {arXiv preprint arXiv:2107.07455},
  year      = {2021},
}
```

If you have any questions about the Shifts Dataset, the paper or the benchmarks, please contact `am969@yandex-team.ru` . 


# Dataset Download And Licenses

## License
The Shifts datasets are released under different license.

### White matter multiple sclerosis lesions segmentation

Data is distributed under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. Data can be downloaded after signing [OFSEP](https://www.ofsep.org/fr/) data usage agreement.

### Marine Cargo Vessel Power Estimation
The data are released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. 
By downloading the data, you are accepting and agreeing to the terms of the [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
license.

The vessel power estimation dataset consists of measurements sampled every minute from sensors on-board a merchant ship over a span of 4 years, 
cleaned and augmented with weather data from a third-party provider. 
We also provide a synthetic benchmark dataset that contains the same splits and input features as in the real data, 
but the target power labels are replaced with predictions of an analytical physics-based vessel model. 

### Weather Prediction

The Shifts Weather Prediction Dataset  is released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. This dataset was constructed by combining features from publicly available weather prediction services and models. Specifically, we combined data from [NOAA/NWS servers](https://www.weather.gov/disclaimer), data generated by WRF model from [NCAR/UCAR](https://github.com/wrf-model/WRF/blob/master/LICENSE.txt),  and data from [Meteorological Service of Canada](https://www.canada.ca/en/transparency/terms.html).  Ground station readings were taken from [NOAA] (https://www.weather.gov/disclaimer). The data was cleaned and features standardized. 

### Machine Translation
  
The Shifts Machine Translation Dataset is released under a mixed license.
  
GlobalVoices evaluation data is released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 
  
The english source data was taken from [GlobalVoices]( https://globalvoices.org) (originally licenced under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/legalcode)) and target Russian translations provided by Yandex in-house professional translators.
  
The source-side text for the Reddit development and evaluation datasets exist under terms of the Reddit API. The target side Russian sentences were obtained by Yandex via in-house professional translators and are released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). We highlight that the development set source sentences are the same ones as used in the [MTNT dataset](http://www.cs.cmu.edu/~pmichel1/mtnt/).

### Motion Prediction
  
Shifts SDC Motion Prediction Dataset is released under [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.

## Download links

**By downloading the Shifts Dataset, you automatically agree to the licenses described above.**

### White matter multiple sclerosis lesions segmentation

Already preprocessed data can be downloaded from [zenodo](https://zenodo.org/record/7051658).
The baseline models can be downloaded from this [link](https://drive.google.com/file/d/1eTTgga7Cd1GjR0YupVbLuLd3unl6_Jj3/view?usp=sharing).
All the code of the baseline models and uncertainty measures is provided in the ["shifts/mswml/"](https://github.com/Shifts-Project/shifts/tree/main/mswml) folder.

### Marine Cargo Vessel Power Estimation
For the synthetic data, the canonical partitions of the training and development data can be downloaded from
[Zenodo](https://zenodo.org/record/7057666#.YzGjddJBw5m). 

The synthetic evaluation and generalization sets and 
the canonical partitions of the real data will be gradually made available based on the timeline of the [Shifts Challenge 2022](https://shifts.grand-challenge.org/).

### Weather Prediction

Canonical parition of the training, development and evaluation data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar). The full dataset can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/weather/full-dataset.tar).  Baseline models can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/weather/baseline-models.tar).

The weather and motion prediction data should be as simple to load as the development data. 

### Machine Translation

The training data for this task is the [WMT'20 En-Ru](http://www.statmt.org/wmt20/translation-task.html) dataset can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/translation/train-data.tar), the development data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/translation/dev-data.tar) and the evaluation data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/translation/eval-data.tar) All data is automatically downloaded via the scripts provided [here](https://github.com/yandex-research/shifts/tree/main/translation). Baseline models can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/translation/baseline-models.tar).

A description of how to process the evaluation data for the translation track is provided [here](https://github.com/yandex-research/shifts/tree/main/translation).

### Motion Prediction

Canonical parition of the training and development data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-trn-dev-data.tar). The canonical parition of the evaluation data can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/canonical-eval-data.tar). The full, unpartitioned dataset is available [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/full-unpartitioned-data.tar). Baseline models can be downloaded [here](https://storage.yandexcloud.net/yandex-research/shifts/sdc/baseline-models.tar).





