import numpy as np
import pandas as pd
import bisect
from sklearn.utils import shuffle

class Config():
    '''
    Define Configuration for partioning data
    '''
    def __init__(self, time_splits = [0.6, 0.1, 0.15, 0.15], climate_splits = [3,1,1], in_domain_splits=[0.7, 0.15, 0.15], seed=1, eval_dev_overlap=True):
        '''
        time_splits: fractions associated with TRAIN, GAP, DEV_OUT, EVAL_OUT (split on time)
        climate_splits: number of climates kept for TRAIN, DEV_OUT, EVAL_OUT (from above time splits)
        in_domain_splits: Separation TRAIN time and specified climate segment block into TRAIN, DEV_IN, EVAL_IN
        eval_dev_overlap: Flag if TRUE, EVAL_OUT climates kept include the DEV_OUT climates kept.
        '''
        self.time_splits = time_splits
        self.climate_splits = climate_splits
        self.in_domain_splits = in_domain_splits
        self.seed = seed
        self.eval_dev_overlap = eval_dev_overlap

        self.run_checks()

    def run_checks(self):
        assert len(self.time_splits) == 4, "Need 4 time splits, e.g. [0.6, 0.15, 0.15, 0.15]"
        assert len(self.climate_splits) ==3, "Need 3 climate splits, e.g. [3,1,1]"
        assert np.sum(np.asarray(self.time_splits)) == 1.0, "Time split fractions should add to one"
        assert np.sum(np.asarray(self.climate_splits)) == 5, "Climate split counts should add to 5"
        assert np.sum(np.asarray(self.in_domain_splits)) == 1.0, "In domain split fractions should add to one"


class Partitioner():
    '''
    Requires a block of data and partitions it into
    the following disjoint subsets:


    1) train.csv:
                    Data for training.
    
    2) dev_in.csv:
                    Development data from the same domain
                    in time and climate as of the train.csv
                    data.

    3) eval_in.csv:
                    Evaluation data from the same domain
                    in time and climate as of the train.csv
                    data.

    4) dev_out.csv:
                    Data distributionally shifted in time and climate
                    from train.csv.

    5) eval_out.csv:
                    Data further distributionally shifted in climate
                    and different time frame from train.csv and dev_out.csv.
                    Can be configured to have overlap in climates
                    with dev_out.csv.

    If no_meta == True, a further set of files will be generated:

    6) dev_in_no_meta.csv:
                    Same as dev_in.csv with meta data (first 6 features)
                    removed.

    7) eval_in_no_meta.csv:
                    Same as eval_in.csv with meta data (first 6 features)
                    removed.

    8) dev_out_no_meta.csv:
                    Same as dev_out.csv with meta data (first 6 features)
                    removed.
    
    9) eval_out_no_meta.csv:
                    Same as eval_out.csv with meta data (first 6 features)
                    removed.

    '''
    def __init__(self, data_path, climate_info_path, config=Config()):

        # Define the 5 climate types
        self.CLIMATES = ['tropical', 'dry', 'mild temperate', 'snow', 'polar']
        self.config = config
        # Read in the raw data
        self.df = pd.read_csv(data_path)
        # Introduce an additional column for the climate type
        self._include_climate(climate_info_path)
        # Partition the data by time segments
        self._split_by_time()
        # Partition the data by climate segments
        self._split_by_climate()
        # Add dummy samples for unrepresented classifcation classes
        self.dfs_to_save['train'] = self._add_dummy(self.dfs_to_save['train'])
        
    
    def _include_climate(self, climate_info_path):
        '''
        Add column with climate type based on location
        '''
        # Define mapping between climate code to climate name
        letter_to_name = {
            'A' : 'tropical',
            'B' : 'dry',
            'C' : 'mild temperate',
            'D' : 'snow',
            'E' : 'polar',
            'n' : 'other'
        }

        # Load data about longitude and latitude to climate type from http://hanschen.org/koppen
        df_climate_info = pd.read_csv(climate_info_path, sep='\t')
        # Get the longitudes and latitudes at every 0.5 degrees resolution on land
        climate_longitudes = list(df_climate_info['longitude'])
        climate_latitudes = list(df_climate_info['latitude'])
        # Identify one of the five climate types using the latest climate type data provided (2010)
        climate_types = [str(typ)[0] for typ in list(df_climate_info['p2010_2010'])]
        # Load the longitudes and latitudes of all raw data
        y_lats = list(self.df['fact_latitude'])
        y_longs = list(self.df['fact_longitude'])
        # Match longitudes and latitudes to closest 0.5 degree resolution to identify climate type
        y_climates = [self._get_climate(lat, long, climate_latitudes, climate_longitudes, climate_types, count) for count, (lat, long) in enumerate(zip(y_lats, y_longs))]
        # Convert climate code names to actual names and add climate information to the raw data
        y_climates = [letter_to_name[clim] for clim in y_climates]
        self.df.insert(5, 'climate', y_climates)      

    def _get_climate(self, lat, long, climate_latitudes, climate_longitudes, climate_types, count):
        """
        Map lat, long to the closest climate_latitude and climate_longitude and then identify
        corresponding climate type.
        """
        # Find index of first occurence greater than the specific longitude
        ind_climate_long_start = bisect.bisect_left(climate_longitudes, long)
        # Find index of first occurence greater than longitude + 0.5
        ind_climate_long_end = bisect.bisect_left(climate_longitudes, long+0.5)
        # Relative (to the longitude index) index to identify the closest latitude index
        rel_ind = len(climate_latitudes[ind_climate_long_start: ind_climate_long_end]) - bisect.bisect_left(climate_latitudes[ind_climate_long_start+1: ind_climate_long_end][::-1], lat) - 1
        # The overall index of the closest latitude, longitude point
        ind = ind_climate_long_start + rel_ind

        # approx_lat, approx_long = climate_latitudes[ind], climate_longitudes[ind]
        # Find the corresponding climate type
        climate = climate_types[ind]
        return climate
    
    def _split_by_time(self):
        """
        Partition the data into the main time segments.
        """
        # Sort all data in time order
        self.df = self.df.sort_values(by=['fact_time'])
        # Find the total number of data points
        num_samples = len(self.df)
        # Use the time fractions to identify the index splits for the raw data
        first_split_ind = int(num_samples*self.config.time_splits[0])
        second_split_ind = int(num_samples*self.config.time_splits[1]) + first_split_ind
        third_split_ind = int(num_samples*self.config.time_splits[2]) + second_split_ind
        # Identify the train segment
        self._df_train_all = self.df.iloc[:first_split_ind]
        # Identify the time segment to be rejected
        self._df_gap_all = self.df.iloc[first_split_ind:second_split_ind]
        # Identify the time segment for the eval_out data
        self._df_eval_all = self.df.iloc[second_split_ind:third_split_ind]
        # Identify the time segment for the dev_out data
        self._df_dev_all = self.df.iloc[third_split_ind:]

    def _split_by_climate(self):

        # Use the climate split fractions to identify the climate types partitions
        clim_first_split_ind = self.config.climate_splits[0]
        clim_second_split_ind = self.config.climate_splits[0] + self.config.climate_splits[1]
        # Identify the climate types to keep for the train data
        train_climates_keep = self.CLIMATES[:clim_first_split_ind]
        df_train_kept_climates = self._df_train_all.loc[self._df_train_all['climate'].isin(train_climates_keep)]
        # Shuffle the in-domain data
        df_train_kept_climates = shuffle(df_train_kept_climates, random_state=self.config.seed)
        # Use the in-domain splits to get splits for train, dev_in and eval_in
        first_train_split_ind = int(len(df_train_kept_climates) * self.config.in_domain_splits[0])
        second_train_split_ind = first_train_split_ind + int(len(df_train_kept_climates) * self.config.in_domain_splits[1])
        # Extract the train data
        df_train = df_train_kept_climates.iloc[:first_train_split_ind]
        # Extract the dev_in data
        df_dev_in_domain = df_train_kept_climates.iloc[first_train_split_ind:second_train_split_ind]
        # Extract the eval_in data
        df_eval_in_domain = df_train_kept_climates.iloc[second_train_split_ind:]
        # Identify the climate types to keep for the dev_out data
        dev_climates_keep = self.CLIMATES[clim_first_split_ind:clim_second_split_ind]
        # Extract the dev_out data
        df_dev = self._df_dev_all.loc[self._df_dev_all['climate'].isin(dev_climates_keep)]
        # Identify the climate types to keep for the eval_out data
        eval_climates_keep = self.CLIMATES[clim_second_split_ind:]
        # Include additional climates in the eval_out set if desired
        if self.config.eval_dev_overlap: eval_climates_keep += dev_climates_keep
        # Extract the eval_out data
        df_eval = self._df_eval_all.loc[self._df_eval_all['climate'].isin(eval_climates_keep)]

        self.dfs_to_save = {}
        self.dfs_to_save['train'] = df_train
        self.dfs_to_save['dev_in'] = df_dev_in_domain
        self.dfs_to_save['eval_in'] = df_eval_in_domain
        self.dfs_to_save['dev_out'] = df_dev
        self.dfs_to_save['eval_out'] = df_eval

    def _add_dummy(self, df_to_modify):
        '''
        Add dummy data for missing precipitation classes in df.
        '''
        # Identify list of all classification classes
        classes_to_check = set(list(self.df['fact_cwsm_class']))
        # Find the average of all data rows
        avg_row = df_to_modify.mean(axis=0)
        # Append averaged row for each classification class not present in the data
        for precip_class in classes_to_check:
            if precip_class not in df_to_modify['fact_cwsm_class']:
                print("Dummy added to training", precip_class)
                ind = len(df_to_modify)
                df_to_modify.loc[ind] = avg_row
                df_to_modify.at[ind, 'fact_cwsm_class'] = precip_class
        return df_to_modify

    def _remove_meta(self, df):
        """
        Remove data that should not be used at training or testing time.
        """
        return df.iloc[:,6:] # first 6 columns (including climate type) are meta data

    def save(self, save_path, no_meta=False):
        """
        Save all relevant data split files.
        """
        if no_meta:
            # Create equivalent files with the meta data removed
            self.dfs_to_save['dev_in_no_meta'] = self._remove_meta(self.dfs_to_save['dev_in'])
            self.dfs_to_save['eval_in_no_meta'] = self._remove_meta(self.dfs_to_save['eval_in'])
            self.dfs_to_save['dev_out_no_meta'] = self._remove_meta(self.dfs_to_save['dev_out'])
            self.dfs_to_save['eval_out_no_meta'] = self._remove_meta(self.dfs_to_save['eval_out'])

        # Save all files
        for name, df in self.dfs_to_save.items():
            df.to_csv(save_path+'/'+name+'.csv', index=False)
            print('Saved', name)


