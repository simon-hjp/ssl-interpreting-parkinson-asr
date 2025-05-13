import os
import torch
import numpy as np
import pandas as pd

# class ParkinsonDataset(torch.utils.data.Dataset):
#     def __init__(self, config, dataset_path, is_training=True):

#         self.config = config
#         self.dataset = pd.read_csv(dataset_path)

#         # -- filtering tasks we are interested in, both for training and evaluation
#         task_filter = [task['name'] for task in config.tasks]
#         self.dataset = self.dataset[self.dataset['task_id'].isin(task_filter)]

#         # -- collecting informed speech feature metadata
#         informed_metadata_ids = []
#         informed_metadata_bounds = {}
#         self.target_informed_idxs = {}
#         for feature in self.config.features:
#             feature_metadata_df = pd.read_csv(feature['metadata'])

#             feature_metadata_ids = feature_metadata_df['feature_id'].tolist()
#             self.target_informed_idxs[feature['name']] = feature_metadata_df['index_pos'].tolist()

#             start_bound = len(informed_metadata_ids)
#             end_bound = start_bound + len(feature_metadata_ids)

#             informed_metadata_ids += feature_metadata_ids
#             informed_metadata_bounds[feature['name']] = (start_bound, end_bound)

#         self.informed_metadata = (informed_metadata_ids, informed_metadata_bounds)

#         # -- median and standard deviation of HC subjects in training for each type of feature.
#         if is_training:
#             # self.feature_norm_stats = self.__compute_feature_norm_stats__(self.dataset[self.dataset['label'] == 0])
#             self.feature_norm_stats = self.__compute_feature_norm_stats__(self.dataset[self.dataset['label'] == 0])
#         else:
#             # WARNING: For validation and test, these statistics are replaced in pipeline.py by those computed for training!
#             self.feature_norm_stats = None

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         sample = self.dataset.iloc[index]

#         batch_sample = {}

#         # -- subject and sample identification
#         batch_sample['subject_id'] = sample['subject_id']
#         batch_sample['sample_id'] = sample['sample_id']

#         # -- speech-based features
#         if self.config.model not in ['self_ssl']:
#             batch_sample['informed_metadata'] = []
#             for feature in self.config.features:
#                 # feature_data = np.load(sample[feature['name']])['data'][:, self.target_informed_idxs[feature['name']]]
#                 feature_data = np.atleast_2d(np.load(sample[feature['name']])['data'])[:, self.target_informed_idxs[feature['name']]]

#                 feature_std = self.feature_norm_stats[feature['name']]['std']
#                 feature_median = self.feature_norm_stats[feature['name']]['median']
#                 batch_sample[feature['name']] = np.divide(feature_data - feature_median, feature_std, out=np.zeros(feature_data.shape), where=feature_std!=0)

#         # -- ssl speech features
#         if self.config.model not in ['self_inf']:
#             # ssl_data = np.load(sample[self.config.ssl_features])['data']
#             ssl_data = np.atleast_2d(np.load(sample[self.config.ssl_features])['data'])
#             ssl_median = self.feature_norm_stats[self.config.ssl_features]['median']
#             ssl_std = self.feature_norm_stats[self.config.ssl_features]['std']

#             batch_sample[self.config.ssl_features] = np.divide(ssl_data - ssl_median, ssl_std, out=np.zeros(ssl_data.shape), where=ssl_std!=0)

#         # -- label ground truth
#         batch_sample['label'] = sample['label']

#         return batch_sample

#     def collate_fn(self, batch):
#         pad_batch = {}

#         for key in batch[0].keys():
#             if key in [self.config.ssl_features]:
#                 pad_batch[key] = [torch.Tensor(batch_sample[key]) for batch_sample in batch]
#                 # -- computing mask
#                 pad_batch['ssl_lengths'] = [ssl_sample.shape[0] for ssl_sample in pad_batch[key]]
#                 pad_batch['mask_ssl'] = (~self.__make_pad_mask__(pad_batch['ssl_lengths'])[:, None, :])
#             else:
#                 pad_batch[key] = [batch_sample[key] for batch_sample in batch]

#             if key not in ['subject_id', 'sample_id', 'group', 'task_id']:
#                 if key in [self.config.ssl_features]:
#                     pad_batch[key] = torch.nn.utils.rnn.pad_sequence(pad_batch[key], batch_first=True).type(torch.float32)
#                 elif key not in ['mask_ssl']:
#                     pad_batch[key] = torch.Tensor(np.array(pad_batch[key])).type(torch.float32 if key not in ['label', 'ssl_lengths'] else torch.int64)

#         return pad_batch

#     def __make_pad_mask__(self, lengths):
#         bs = int(len(lengths))
#         maxlen = int(max(lengths))

#         seq_range = torch.arange(0, maxlen, dtype=torch.int64)
#         seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
#         seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
#         mask = seq_range_expand >= seq_length_expand

#         return mask

#     def __compute_feature_norm_stats__(self, hc_dataset):
#         features_ids = [feature['name'] for feature in self.config.features] + [self.config.ssl_features]
#         feature_norm_stats = {feature_id:{'median': 0.0, 'std': 1.0} for feature_id in features_ids}

#         # -- statistics for informed speech features
#         if self.config.model not in ['self_ssl']:
#             for feature in self.config.features:

#                 # samples = np.array([
#                 #     np.load(sample_path)['data'][:, self.target_informed_idxs[feature['name']]]
#                 #     for sample_path in hc_dataset[feature['name']].tolist()
#                 # ])

#                 # feature_norm_stats[feature['name']]['median'] = np.median(samples, axis=0)
#                 # feature_norm_stats[feature['name']]['std'] = np.std(samples, axis=0)
#                 samples = [
#                     np.atleast_2d(np.load(p)['data'])[:, self.target_informed_idxs[feature['name']]]
#                     for p in hc_dataset[feature['name']].tolist()
#                 ]
#                 concat = np.concatenate(samples, axis=0)
#                 feature_norm_stats[feature['name']]['median'] = np.median(concat, axis=0)
#                 feature_norm_stats[feature['name']]['std'] = np.std(concat, axis=0)

#         # -- statistics for SSL speech features
#         if self.config.model not in ['mlp_inf', 'self_inf']:
#             # samples = np.concatenate([
#             #     np.load(sample_path)['data']
#             #     for sample_path in hc_dataset[self.config.ssl_features].tolist()
#             # ], axis=0 )
#             samples = np.concatenate([
#                 np.atleast_2d(np.load(p)['data'])
#                 for p in hc_dataset[self.config.ssl_features].tolist()
#             ], axis=0)

#             feature_norm_stats[self.config.ssl_features]['median'] = np.median(samples, axis=0)
#             feature_norm_stats[self.config.ssl_features]['std'] = np.std(samples, axis=0)

#         return feature_norm_stats


class ParkinsonDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_path, is_training=True):
        self.config = config
        self.dataset = pd.read_csv(dataset_path)

        # Filter tasks
        task_filter = [task['name'] for task in config.tasks]
        self.dataset = self.dataset[self.dataset['task_id'].isin(task_filter)]

        # Informed features
        informed_metadata_ids = []
        informed_metadata_bounds = {}
        self.target_informed_idxs = {}
        for feature in self.config.features:
            feature_metadata_df = pd.read_csv(feature['metadata'])
            feature_metadata_ids = feature_metadata_df['feature_id'].tolist()
            self.target_informed_idxs[feature['name']] = feature_metadata_df['index_pos'].tolist()
            start_bound = len(informed_metadata_ids)
            end_bound = start_bound + len(feature_metadata_ids)
            informed_metadata_ids += feature_metadata_ids
            informed_metadata_bounds[feature['name']] = (start_bound, end_bound)

        self.informed_metadata = (informed_metadata_ids, informed_metadata_bounds)

        # Support multiple SSL features
        self.ssl_feature_names = config.ssl_features if isinstance(config.ssl_features, list) else [config.ssl_features]

        if is_training:
            self.feature_norm_stats = self.__compute_feature_norm_stats__(self.dataset[self.dataset['label'] == 0])
        else:
            self.feature_norm_stats = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]
        batch_sample = {
            'subject_id': sample['subject_id'],
            'sample_id': sample['sample_id']
        }

        # Informed features
        if self.config.model not in ['self_ssl']:
            batch_sample['informed_metadata'] = []
            for feature in self.config.features:
                feature_data = np.atleast_2d(np.load(sample[feature['name']])['data'])[:, self.target_informed_idxs[feature['name']]]
                std = self.feature_norm_stats[feature['name']]['std']
                median = self.feature_norm_stats[feature['name']]['median']
                batch_sample[feature['name']] = np.divide(feature_data - median, std, out=np.zeros_like(feature_data), where=std != 0)

        # SSL features
        if self.config.model not in ['self_inf']:
            for ssl_feat in self.ssl_feature_names:
                ssl_data = np.atleast_2d(np.load(sample[ssl_feat])['data'])
                std = self.feature_norm_stats[ssl_feat]['std']
                median = self.feature_norm_stats[ssl_feat]['median']
                batch_sample[ssl_feat] = np.divide(ssl_data - median, std, out=np.zeros_like(ssl_data), where=std != 0)

        batch_sample['label'] = sample['label']
        return batch_sample

    def __compute_feature_norm_stats__(self, hc_dataset):
        feature_ids = [f['name'] for f in self.config.features] + self.ssl_feature_names
        stats = {fid: {'median': 0.0, 'std': 1.0} for fid in feature_ids}
        print(hc_dataset.columns)

        # Informed
        if self.config.model not in ['self_ssl']:
            for feature in self.config.features:
                print(feature, '----------------------------------------')
                samples = [
                    np.atleast_2d(np.load(p)['data'])[:, self.target_informed_idxs[feature['name']]]
                    for p in hc_dataset[feature['name']].tolist()
                ]
                concat = np.concatenate(samples, axis=0)
                stats[feature['name']]['median'] = np.median(concat, axis=0)
                stats[feature['name']]['std'] = np.std(concat, axis=0)

        # SSL
        if self.config.model not in ['mlp_inf', 'self_inf']:
            for ssl_feat in self.ssl_feature_names:
                samples = np.concatenate([
                    np.atleast_2d(np.load(p)['data'])
                    for p in hc_dataset[ssl_feat].tolist()
                ], axis=0)
                stats[ssl_feat]['median'] = np.median(samples, axis=0)
                stats[ssl_feat]['std'] = np.std(samples, axis=0)

        return stats

    def collate_fn(self, batch):
        pad_batch = {}

        for key in batch[0].keys():
            if key in self.ssl_feature_names:
                pad_batch[key] = [torch.Tensor(batch_sample[key]) for batch_sample in batch]
                
                # -- computing mask
                pad_batch['ssl_lengths'] = [ssl_sample.shape[0] for ssl_sample in pad_batch[key]]
                pad_batch['mask_ssl'] = (~self.__make_pad_mask__(pad_batch['ssl_lengths'])[:, None, :])
            else:
                pad_batch[key] = [batch_sample[key] for batch_sample in batch]

            if key not in ['subject_id', 'sample_id', 'group', 'task_id']:
                if key in self.ssl_feature_names:
                    pad_batch[key] = torch.nn.utils.rnn.pad_sequence(pad_batch[key], batch_first=True).type(torch.float32)
                elif key not in ['mask_ssl']:
                    pad_batch[key] = torch.Tensor(np.array(pad_batch[key])).type(torch.float32 if key not in ['label', 'ssl_lengths'] else torch.int64)

        return pad_batch


    def __make_pad_mask__(self, lengths):
        bs = len(lengths)
        maxlen = max(lengths)
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        lengths_expand = seq_range_expand.new_tensor(lengths).unsqueeze(-1)
        return seq_range_expand >= lengths_expand
