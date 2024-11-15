import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List  # used for type hinting
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE


class Data:

    def __init__(self):
        """
        # self.data :standard dataframe
        # self.columns: list of headers of dataframe
        # self.padded_dataframe: SORTED dataframe with padded values (on labeled data only!)
        # self.binary_data: dataframe with 0's for missing values and 1's for non-missing values
        # self.parquet_data_dictionary: dictionary with id as a key and 2d accelerometer data as value
        # self.ids_list: list with all ids *WITH ACCELEROMETER DATA*
        # self.ids_labels: dictionary with id, label pair *ONLY OF ACCELEROMETERED ID'S*
        """

        self.padded_dataframe = pd.read_excel('sorted_padded_data_.xlsx')
        root_dir = 'series_train.parquet'
        # Read the Parquet file
        df = pd.read_csv('train.csv')

        # Preview the  first few rows
        self.columns: List[str] = df.columns
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
        self.data: pd.DataFrame = df
        self.binary_data: pd.DataFrame = binary_df

        parquet_data_dictionary = {}
        ids_list = []
        ids_labels = {}

        # iterating through each subdirectory
        # for id_folder in tqdm(os.listdir(root_dir), desc="Reading .parquet files...", unit="item", ncols=100, colour="blue"):
        #     id_folder_path = os.path.join(root_dir, id_folder)
        #     if os.path.isdir(id_folder_path):
        #         for file in os.listdir(id_folder_path):
        #             file_path = os.path.join(id_folder_path, file)
        #
        #             id_ = file_path[-23:-15]  # unique id of current file
        #             df_: pd.DataFrame = pd.read_parquet(file_path)
        #
        #             # replacing steps (which is same as index) to label
        #             df_.rename(columns={'step': 'sii'}, inplace=True)
        #
        #             dataset_id_row = self.data.loc[self.data['id'] == id_]
        #             current_id_label = dataset_id_row['sii'].iloc[0]
        #
        #             ids_labels[id_] = current_id_label
        #             ids_list.append(id_)
        #             df_['sii'] = current_id_label
        #
        #             # adding dataframe to dictionary
        #             parquet_data_dictionary[id_] = df_

        # key: id, value: 2d tabular accelerometer data
        self.parquet_data_dictionary: Dict[str, pd.Dataframe] = parquet_data_dictionary
        self.ids_list: List[str] = ids_list
        self.ids_labels: Dict[str, float] = ids_labels

    def get_binary_heatmap(self):
        '''
        plots binary heatmap showing absence of data in all columns in regular dataframe.
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

    def get_correlation_heatmap(self):
        '''

        :param dataframe: 2d pandas dataframe
        :return: plots correlation heatmap of given dataframe.
        '''

        dataframe = self.data
        # cant plot id's
        try:
            dataframe.drop(columns='id', inplace=True)
        except KeyError:
            pass
        correlation_matrix = dataframe.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def get_mean_column_on_accelerometer_data(self, id_: str, column: str) -> float:
        """

        :param id_: unique ID of person with accelerometer data
        :param column: column to get mean value from

        :return: mean value of column
        """
        dataframe: pd.DataFrame = self.parquet_data_dictionary[id_]
        return dataframe[column].mean()

    def get_violin_graph_of_mean_enmos_per_label(self):

        mean_enmos = []
        labels = []
        for id_ in self.ids_list:
            mean_enmos.append(self.get_mean_column_on_accelerometer_data(id_, 'enmo'))
            labels.append(self.ids_labels[id_])

        foo = pd.DataFrame({
            'Values': mean_enmos,
            'Category': labels
        })

        sns.violinplot(x='Category', y='Values', data=foo, inner='point', scale='width', bw=0.2)

        # Set x-axis limits as specified
        plt.ylim(0, 0.2)
        plt.xlabel('Internet addiction test result (label)')
        plt.ylabel('Values')
        plt.title(r"Distribution of mean ENMO's by internet addiction test result")

        plt.show()

    def get_violing_graph_of_mean_light(self):

        mean_light = []
        labels = []
        for id_ in self.ids_list:
            mean_light.append(self.get_mean_column_on_accelerometer_data(id_, 'light'))
            labels.append(self.ids_labels[id_])
        foo = pd.DataFrame({
            'Values': mean_light,
            'Category': labels
        })

        sns.violinplot(x='Category', y='Values', data=foo, inner='point', scale='width', bw=0.2)

        # Set x-axis limits as specified
        plt.ylim(0, max(mean_light))
        plt.xlabel('Internet addiction test result (label)')
        plt.ylabel('Values')
        plt.title(r"Distribution of mean light by internet addiction test result")

        plt.show()

    @staticmethod
    def get_kmeans_fit_unlabeled_data(file_path='sorted_padded_data_.xlsx') -> pd.DataFrame:
        """

        :param file_path
        :return: returns pd Dataframe of given data and predicted sii on sii column, using kmeans clustering with
        euclidian distance from centroids on non-missing data.
        """
        file_path = 'sorted_padded_data_.xlsx'
        df = pd.read_excel(file_path)

        # dropping id column for clustering
        ids = df['id']
        df = df.drop(columns=['id'])

        # features
        features = df.columns.difference(['sii'])

        # ensuring data conversing
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')

        labeled_data = df[df['sii'].notna()]
        unlabeled_data = df[df['sii'].isna()]

        # training kmeans on labeled data with non-missing values
        labeled_features = labeled_data[features].dropna()
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(labeled_features)

        # custom distances using available features only
        def custom_distance(point, centroids):
            point = np.array(point, dtype=np.float64)
            available_features = ~np.isnan(point)
            if np.sum(available_features) == 0:  # return infinity if all values are NaN
                return np.inf * np.ones(centroids.shape[0])
            # take euclidian distance from centroids of clusters
            return cdist([point[available_features]], centroids[:, available_features], metric='euclidean')[0]

        # predicting clusters for unlabeled data using the custom distance function
        predicted_labels = []
        for _, row in unlabeled_data.iterrows():
            distances = custom_distance(row[features].values, kmeans.cluster_centers_)
            predicted_label = np.argmin(distances) if np.isfinite(distances).any() else -1
            predicted_labels.append(predicted_label)

        unlabeled_data['sii'] = predicted_labels
        labeled_data: pd.DataFrame = unlabeled_data

        return labeled_data

    def smote_oversample_label_3_data(self) -> pd.DataFrame:
        """

        :return: pd dataframe of new generated entries for sii=3.
        returns 159 entries (10% of majority class - sii=0)
        """
        df = self.padded_dataframe
        df = df.drop('id', axis=1)

        label_3_data = df[df['sii'] == 3].drop('sii', axis=1)
        label_3_labels = df[df['sii'] == 3]['sii']

        other_data = df[df['sii'] == 0].drop('sii', axis=1)
        other_labels = df[df['sii'] == 0]['sii']

        # aombine data for label '3' and the other class 0 - majority class
        combined_data = pd.concat([label_3_data, other_data])
        combined_labels = pd.concat([label_3_labels, other_labels])

        # apply SMOTE to oversample label '3' class
        smote = SMOTE(sampling_strategy=0.1,
                      random_state=42)  # generating 159 samples of label 3 data (10% of majority class)
        x_resampled, y_resampled = smote.fit_resample(combined_data, combined_labels)

        resampled_df = pd.DataFrame(x_resampled, columns=combined_data.columns)
        resampled_df['sii'] = y_resampled

        df = resampled_df[resampled_df['sii'] == 3]
        df = df.reset_index(drop=True)

        df['id'] = 'smote_generated_entry'

        return df


dataset = Data()
# dataset.get_binary_heatmap()                          # to get heatmap of (regular) dataset
# print(dataset.columns)                                # to get all columns list
# dataset.get_column_data_percentage('sii')             # to get data absense percentage on parameter column
# dataset.get_correlation_heatmap()                     # to get correlation heatmap
# dataset.get_violin_graph_of_mean_enmos_per_label()    # to get mean enmos and label violin plot
# dataset.get_violin_graph_of_mean_light()              # to get mean light and label violin plot
# Data.get_kmeans_fit_unlabeled_data()                  # to label the unlabeled data using kmeans clustering
# print(dataset.smote_oversample_label_3_data())        # to get SMOTE oversampled data from sii=3
#######################################
