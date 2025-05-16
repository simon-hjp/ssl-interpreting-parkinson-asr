# import os
# import pickle
# import argparse
# import numpy as np
# import pandas as pd
# from sklearn.metrics import classification_report

# def compute_average_report_across_runs(reports):
#     overall_report = {
#         'HC': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'PD': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'accuracy': [],
#         'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#     }

#     # Spoiler: It's gonna be inefficient :)

#     for report in reports:
#         for key in report.keys():
#             if key == 'accuracy':
#                 overall_report[key].append(report[key])
#             else:
#                 for key2 in report[key].keys():
#                     overall_report[key][key2].append(report[key][key2])

#     for key in overall_report.keys():
#             if key == 'accuracy':
#                 overall_report[key] = f'{round(np.array(overall_report[key]).mean(), 4)}±{round(np.array(overall_report[key]).std(), 4)}'
#             else:
#                 for key2 in report[key].keys():
#                     overall_report[key][key2] = f'{round(np.array(overall_report[key][key2]).mean(), 4)}±{round(np.array(overall_report[key][key2]).std(), 4)}'

#     # -- just for a more clean output
#     overall_report = pd.DataFrame.from_dict(overall_report).T
#     overall_report.iloc[2,0] = ''
#     overall_report.iloc[2,1] = ''
#     overall_report.iloc[2,3] = overall_report.iloc[3,3]

#     return overall_report

# def get_reports(exps_dir):
#     val_reports = []
#     test_reports = []
#     run_dirs = os.listdir(exps_dir)
#     for run_dir in run_dirs:
#         val_preds = []
#         val_labels = []
#         test_preds = []
#         test_labels = []

#         run_dir_path = os.path.join(exps_dir, run_dir)
#         fold_dirs = os.listdir(run_dir_path)
#         for fold_dir in fold_dirs:
#             model_output_dir = os.path.join(run_dir_path, fold_dir, 'model_output')

#             # -- validation set
#             val_report_path = os.path.join(os.path.join(model_output_dir, 'validation_classification.pkl'))
#             with open(val_report_path, 'rb') as f:
#                 val_model_output = pickle.load(f)

#             val_preds += val_model_output['preds']
#             val_labels += val_model_output['labels']

#             # -- test set
#             test_report_path = os.path.join(os.path.join(model_output_dir, 'test_classification.pkl'))
#             with open(test_report_path, 'rb') as f:
#                 test_model_output = pickle.load(f)

#             test_preds += test_model_output['preds']
#             test_labels += test_model_output['labels']

#         # -- computing reports
#         val_reports.append(
#             classification_report(
#                 val_labels,
#                 val_preds,
#                 target_names=['HC', 'PD'],
#                 output_dict=True,
#             )
#         )

#         test_reports.append(
#             classification_report(
#                 test_labels,
#                 test_preds,
#                 target_names=['HC', 'PD'],
#                 output_dict=True,
#             )
#         )

#     return val_reports, test_reports

# if __name__ == '__main__':

#     # -- command-line arguments
#     parser = argparse.ArgumentParser(description='Overall performance across the multiple assesing fold splits.',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored. It should include each directory for each run.')

#     args = parser.parse_args()

#     val_reports, test_reports = get_reports(args.exps_dir)

#     val_overall_report = compute_average_report_across_runs(val_reports)
#     test_overall_report = compute_average_report_across_runs(test_reports)
#     print('', '-'*21, '\nVALIDATION SET\n', '-'*21, '\n', val_overall_report)

#     print('\n'*3, '-'*21, '\nTEST SET\n', '-'*21, '\n', test_overall_report)

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def compute_average_report_across_runs(reports):
    overall_report = {
        'HC': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'PD': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'accuracy': [],
        'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    }

    # Spoiler: It's gonna be inefficient :)

    for report in reports:
        for key in report.keys():
            if key == 'accuracy':
                overall_report[key].append(report[key])
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2].append(report[key][key2])

    for key in overall_report.keys():
            if key == 'accuracy':
                overall_report[key] = f'{round(np.array(overall_report[key]).mean(), 4)}±{round(np.array(overall_report[key]).std(), 4)}'
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2] = f'{round(np.array(overall_report[key][key2]).mean(), 4)}±{round(np.array(overall_report[key][key2]).std(), 4)}'

    # -- just for a more clean output
    overall_report = pd.DataFrame.from_dict(overall_report).T
    overall_report.iloc[2,0] = ''
    overall_report.iloc[2,1] = ''
    overall_report.iloc[2,3] = overall_report.iloc[3,3]

    return overall_report

def get_reports(exps_dir):
    val_reports = []
    test_reports = []
    run_dirs = os.listdir(exps_dir)
    for run_dir in run_dirs:
        if 'DS' not in run_dir:
            val_preds = []
            val_labels = []
            test_preds = []
            test_labels = []

            run_dir_path = os.path.join(exps_dir, run_dir)
            fold_dirs = os.listdir(run_dir_path)
            print('fold = ' ,fold_dirs)
            for fold_dir in fold_dirs:
                if 'seed' in fold_dir:
                    seed_dir_path = os.path.join(run_dir_path,fold_dir)
                    seed_dirs = os.listdir(seed_dir_path)
                    print('seed = ', seed_dirs)

                    for seed_dir in seed_dirs:
                        if 'fold' in seed_dir:
                            model_output_dir = os.path.join(run_dir_path, fold_dir, seed_dir , 'model_output')

                            # -- validation set
                            val_report_path = os.path.join(os.path.join(model_output_dir, 'validation_classification.pkl'))
                            with open(val_report_path, 'rb') as f:
                                val_model_output = pickle.load(f)

                            val_preds += val_model_output['preds']
                            val_labels += val_model_output['labels']

                            # -- test set
                            test_report_path = os.path.join(os.path.join(model_output_dir, 'test_classification.pkl'))
                            with open(test_report_path, 'rb') as f:
                                test_model_output = pickle.load(f)

                            test_preds += test_model_output['preds']
                            test_labels += test_model_output['labels']

        # -- computing reports
        val_reports.append(
            classification_report(
                val_labels,
                val_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

        test_reports.append(
            classification_report(
                test_labels,
                test_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

    return val_reports, test_reports

if __name__ == '__main__':

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Overall performance across the multiple assesing fold splits.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored. It should include each directory for each run.')

    args = parser.parse_args()

    val_reports, test_reports = get_reports(args.exps_dir)

    val_overall_report = compute_average_report_across_runs(val_reports)
    test_overall_report = compute_average_report_across_runs(test_reports)
    print('', '-'*21, '\nVALIDATION SET\n', '-'*21, '\n', val_overall_report)

    print('\n'*3, '-'*21, '\nTEST SET\n', '-'*21, '\n', test_overall_report)

