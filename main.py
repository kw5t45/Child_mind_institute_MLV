import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Data:

    def __init__(self):
        """
        #self.data :standard dataframe
        #self.columns: list of headers of dataframe
        #self.binary_data: dataframe with 0's for missing values and 1's for non-missing values
        #self.parquet_data_dictionary: dictionary with id as a key and 2d accelerometer data as value

        """

        root_dir = 'series_train.parquet'
        # Read the Parquet file
        df = pd.read_csv('train.csv')

        # Preview the  first few rows
        self.data = df
        self.columns = df.columns
        # mapping strings to numbers
        season_mapping = {
            'Winter': 1,
            'Spring': 2,
            'Summer': 3,
            'Fall': 4
        }
        df = df.replace(season_mapping)

        # binary mapping
        binary_df = df.applymap(lambda x: 1 if pd.notna(x) and pd.to_numeric(x, errors='coerce') is not np.nan else 0)
        self.binary_data = binary_df

        parquet_data_dictionary = {}

        # iterating through each subdirectory
        for id_folder in os.listdir(root_dir):
            id_folder_path = os.path.join(root_dir, id_folder)
            if os.path.isdir(id_folder_path):
                for file in os.listdir(id_folder_path):
                    file_path = os.path.join(id_folder_path, file)

                    id_ = file_path[-23:-15] # unique id of current file
                    df_: pd.DataFrame = pd.read_parquet(file_path)

                    # replacing steps (which is same as index) to label
                    df_.rename(columns={'step': 'sii'}, inplace=True)

                    dataset_id_row = self.data.loc[self.data['id'] == id_]
                    current_id_label = dataset_id_row['sii'].iloc[0]


                    df_['sii'] = current_id_label

                    # adding dataframe to dictionary
                    parquet_data_dictionary[id_] = df_.to_numpy()

        # key: id, value: 2d tabular accelerometer data
        self.parquet_data_dictionary = parquet_data_dictionary

    def get_binary_heatmap(self):
        '''
        plots binary heatmap showing absence of data in all columns in regular dataframe.
        :return:
        '''
        data = self.binary_data
        df = pd.DataFrame(data, columns=self.columns)
        plt.figure(figsize=(20, 15))  # Increase figure size for better readability
        # Default heatmap
        p1 = sns.heatmap(df, cmap='binary', vmin=1, vmax=0)  # vmin and vmax set the color scale from 0 to 1
        plt.show()

    def get_column_data_percentage(self, col: str) -> float:
        '''

        :param col: header name of column of dataframe
        :return: percentage showing absense of data in given column (0-100)
        '''

        ones = 0
        zeros = 0
        for i in (self.binary_data[col]):
            if i == 1:
                ones += 1
            else:
                zeros += 1
        return ones * 100 / (ones + zeros)


dataset = Data()
# dataset.get_binary_heatmap() # to get heatmap of (regular) dataset
# print(dataset.columns) to get all columns list
# dataset.get_column_data_percentage('sii') # to get data absense percentage on parameter column


