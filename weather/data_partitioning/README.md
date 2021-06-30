# Data Partition

## Dependencies

### From PyPI
pip install numpy <br />
pip install pandas <br />
pip install scikit-learn <br />

## Usage

### Example command

```
python run_partition.py /path/to/weather/data/df_multiclass_temperature_latlons.csv /path/to/climate/info/koppen_interannual_1901-2010.tsv /path/to/save/generated/files --time_splits 0.6 0.1 0.15 0.15 --climate_splits 3 1 1 --in_domain_splits 0.836634 0.013366 0.15 --no_meta 'yes' --eval_dev_overlap 'yes'
```
Generate files:
`train.csv` `dev_in.csv` `eval_in.csv` `dev_out.csv` `eval_out.csv` `dev_in_no_meta.csv` `eval_in_no_meta.csv` `dev_out_no_meta.csv` `eval_out_no_meta.csv`

With the configuration parameters specified in the example command above, expect the following file sizes:
 - `train.csv`: 3,129,592
 - `dev_in.csv` : 50,000
 - `eval_in.csv` : 561,105
 - `dev_out.csv` : 50,000 (after subsampling from 524,896)
 - `eval_out.csv` : 576,626

### Splits

![alt text](https://github.com/yandex-research/uncertainty-challenge/blob/0ee9faa49d25a484c15adc893e174f90c4728d38/tabular_weather_prediction/data_partitioning/splits.PNG)

### Description

The block of weather data is partitioned into the following disjoint subsets:

#### 1. `train.csv`
Data for training.

#### 2. `dev_in.csv`
Development data from the same domain in time and climate as of the `train.csv` data.

#### 3. `eval_in.csv`
Evaluation data from the same domain in time and climate as of the `train.csv` data.

#### 4. `dev_out.csv`
Data distributionally shifted in time and climate from `train.csv`.

#### 5. `eval_out.csv`
Data further distributionally shifted in climate and different time frame from `train.csv` and `dev_out.csv`. Can be configured to have overlap in climates with `dev_out.csv`. <br /><br />

If `no_meta == 'yes'`, a further 4 files will be generated:

#### 6. `dev_in_no_meta.csv`
Same as `dev_in.csv` with meta data (first 6 features including climate type) removed.

#### 7. `eval_in_no_meta.csv`
Same as `eval_in.csv` with meta data (first 6 features including climate type) removed.

#### 8. `dev_out_no_meta.csv`
Same as `dev_out.csv` with meta data (first 6 features including climate type) removed.

#### 9. `eval_out_no_meta.csv`
Same as `eval_out.csv` with meta data (first 6 features including climate type) removed.
