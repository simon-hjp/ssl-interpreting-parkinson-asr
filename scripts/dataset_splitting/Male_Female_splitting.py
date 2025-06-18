import pandas as pd
import os
from scripts.dataset_splitting.cross_validation_splitting import stratified_dataset_splitting

def make_single_gender_splits(input_path, output_path, gender = "male"):
    output_path = "./splits/only_m/dataset.csv"
    input_path = "./splits/mixed/dataset.csv"
    full_df = pd.read_csv(input_path)
    if gender == "male":
        gender_df = full_df[full_df["sex"] == 1]
    else:
        gender_df = full_df[full_df["sex"] == 0]
    gender_df.to_csv(output_path)
    stratified_dataset_splitting(input_path, 5)


def split_test_into_two(input_folder):
    input_folder = "./splits/mixed"
    folds = os.listdir(input_folder)
    for fold in folds:
        if "fold" in fold:
            df = pd.read_csv(input_folder + "/" + fold + "/test.csv")
            df_male = df[df["sex"] == 0]
            df_female = df[df["sex"] == 1]
            df_male.to_csv(input_folder + "/" + fold + "/male_test.csv")
            df_female.to_csv(input_folder + "/" + fold + "/female_test.csv")



if __name__ == "__main__":
    # Change all paths below to be correct

    path_to_full_dataset = "./splits/mixed/dataset.csv"
    path_to_folder = "./splits/mixed"

    make_single_gender_splits(path_to_full_dataset, "./splits/only_m/dataset.csv", gender="male")
    make_single_gender_splits(path_to_full_dataset, "./splits/only_f/dataset.csv", gender="female")
    split_test_into_two(path_to_folder)