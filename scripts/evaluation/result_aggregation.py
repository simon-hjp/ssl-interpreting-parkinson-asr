import pandas as pd
import os
import numpy as np

def make_and_save_table(input_path, outputh_path):
    runs_dirs = os.listdir(input_path)
    data = {}
    for run in runs_dirs:
        acc = 0
        for task in os.listdir(input_path + "/" + run):
            for seed in os.listdir(input_path + "/" + run + "/" + task):
                total = 0
                for fold in os.listdir(input_path + "/" + run + "/" + task + "/" + seed):
                    for folder in os.listdir(input_path + "/" + run + "/" + task + "/" + seed + "/" + fold):
                        if folder == "model_output":
                            acc = pd.read_pickle(input_path + "/" + run + "/" + task + "/" + seed + "/" + fold + "/" + folder + "/test_classification.pkl")["acc"]

                    total += acc/5

                if task in data:
                    if run in data[task]:
                        data[task][run].append(total)
                    else:
                        data[task][run] = [total]
                else:
                    data[task] = {run : [total]}

    data = pd.DataFrame(data).reset_index().rename(columns={'index': 'Data'})
    data = data.melt(id_vars='Data', var_name='Task', value_name='Score')
    data["f1"] = [str(np.round(np.mean(scores),2)) + " \pm " + str(np.round(np.std(scores),2)) for scores in data["Score"]]

    print(data)

    table = data[["Data", "Task", "f1"]]
    pivot_df = table.pivot(index='Data', columns='Task', values='f1').reset_index()
    print(pivot_df)

    pivot_df.to_csv(outputh_path)


if __name__ == "__main__":
    # Change all paths below to be correct

    input_path = "../../exps/gita"
    output_path = "./results_table.csv"
    make_and_save_table(input_path, output_path)
