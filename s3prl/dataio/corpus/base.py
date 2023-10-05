import abc

import pandas as pd


class Corpus:
    @property
    @abc.abstractmethod
    def all_data(self) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_split_ids(self):
        raise NotImplementedError

    @property
    def data_split(self):
        train_ids, valid_ids, test_ids = self.data_split_ids
        all_data = self.all_data
        train_data = {idx: all_data[idx] for idx in train_ids}
        valid_data = {idx: all_data[idx] for idx in valid_ids}
        test_data = {idx: all_data[idx] for idx in test_ids}
        return train_data, valid_data, test_data

    @staticmethod
    def dataframe_to_datapoints(df: pd.DataFrame, unique_name_fn: callable):
        data_points = {}
        for _, row in df.iterrows():
            data_point = dict()
            for name, value in row.iteritems():
                data_point[name] = value
            unique_name = unique_name_fn(data_point)
            data_points[unique_name] = data_point
        assert len(data_points) == len(df), f"{len(data_point)} != {len(df)}"
        return data_points
