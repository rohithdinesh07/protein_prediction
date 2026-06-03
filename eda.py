"""
eda.py
Rohith Dinesh
CSE 163

This module performs exploratory data analysis on a protein dataset
derived from UniProt, investigating how amino acid composition and
biochemical properties vary across enzymes, receptors, and structural
proteins.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import (load_dataset, calculate_biochemical_features, sort_data_func_groups)


def dataframe_shape(data: pd.DataFrame) -> None:
    """
    Prints the number of rows and columns in the dataset.

    Args:
        data: The protein DataFrame.
    """
    rows = len(data.index.values)
    columns = len(data.columns)
    print(f"shape: {rows} rows * {columns} columns")


def null_data(data: pd.DataFrame) -> None:
    """
    Prints a summary of missing values in the dataset, including
    per-column NaN counts and overall null percentage.

    Args:
        data: The protein DataFrame.
    """
    print(f"Summary of NaN values: \n\n{data.isnull().sum()}\n")
    total_box = len(data.index.values) * len(data.columns)
    na_total = data.isnull().sum().sum()
    print(f"Null percent: {(na_total / total_box) * 100}%")


def individual_summaries(data: pd.DataFrame) -> None:
    """
    Prints a seven-number summary for all numeric columns in the dataset.

    Args:
        data: The protein DataFrame.
    """
    print(data.select_dtypes(include='number').describe())


def grouped_summaries(data: pd.DataFrame) -> None:
    """
    Prints a seven-number summary for numeric columns grouped by
    functional group (Enzyme, Receptor, Structural).

    Args:
        data: The protein DataFrame containing a 'functional_group' column.
    """
    for group in data['functional_group'].unique():
        print(f"{group} Grouped Data")
        print(data[data['functional_group'] == group].select_dtypes(include='number').describe())


def aa_groups_boxplot(data: pd.DataFrame) -> None:
    """
    Creates and saves a 2x3 grid of boxplots showing the distribution
    of each biochemical feature across the three functional groups.

    Args:
        data: The protein DataFrame with biochemical feature columns.
    """
    features = ["hydrophobic_ratio", "charged_pos_ratio", "charged_neg_ratio",
                "charge_balance", "polar_ratio", "aromatic_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Biochemical Feature Distributions by Functional Group", fontsize=16)
    count = 0
    for i in range(2):
        for j in range (3):
            sns.boxplot(data=data, x='functional_group', y=features[count], ax=axes[i][j], hue='functional_group')
            axes[i][j].set_title(features[count])
            axes[i][j].set_xlabel("Functional Group")
            axes[i][j].set_ylabel(features[count])
            count += 1
    plt.tight_layout()
    plt.savefig('functional_group_boxplot.png')


def plot_heatmap(data: pd.DataFrame) -> None:
    """
    Creates and saves a heatmap of mean biochemical property values
    grouped by functional group.

    Args:
        data: The protein DataFrame with biochemical feature columns.
    """
    prop_cols = ["hydrophobic_ratio", "charged_pos_ratio", "charged_neg_ratio",
                 "charge_balance", "polar_ratio", "aromatic_ratio"]
    heatmap_data = data.groupby('functional_group')[prop_cols].mean()
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Blues")
    plt.title("Average Biochemical Properties by Functional Group")
    plt.tight_layout()
    plt.savefig('aa_group_freq_heatmap.png')


def plot_histograms_func_type(enzyme: pd.DataFrame, receptor: pd.DataFrame,
                              structural: pd.DataFrame) -> None:
    """
    Creates and saves a 3x6 grid of histograms showing the distribution
    of each biochemical feature for each functional group with shared x-axes.

    Args:
        enzyme: DataFrame containing only enzyme proteins.
        receptor: DataFrame containing only receptor proteins.
        structural: DataFrame containing only structural proteins.
    """
    features = ["hydrophobic_ratio", "polar_ratio", "charged_pos_ratio",
            "charged_neg_ratio", "charge_balance", "aromatic_ratio"]

    x_limits = {
        "hydrophobic_ratio": (0.25, 0.65),
        "polar_ratio": (0.10, 0.35),
        "charged_pos_ratio": (0.05, 0.28),
        "charged_neg_ratio": (0.025, 0.225),
        "charge_balance": (-0.08, 0.15),
        "aromatic_ratio": (0.02, 0.18),
    }
    
    groups = [enzyme, receptor, structural]
    group_names = ["Enzyme", "Receptor", "Structural"]
    
    fig, axes = plt.subplots(3, 6, figsize=(30, 11))
    fig.suptitle("Biochemical Feature Distributions by Functional Group", fontsize=16)
    
    for i in range(3):
        for j in range(6):
            sns.histplot(data=groups[i], x=features[j], ax=axes[i][j])
            axes[i][j].set_title(f'{group_names[i]} - {features[j].replace("_", " ").title()}')
            axes[i][j].set_xlabel(features[j])
            axes[i][j].set_ylabel('Count')
            axes[i][j].set_xlim(x_limits[features[j]])

    plt.tight_layout()
    plt.savefig('functional_type_histplots.png')


def size_hist(enzyme: pd.DataFrame, receptor: pd.DataFrame,
              structural: pd.DataFrame) -> None:
    """
    Creates and saves a 3x1 grid of histograms showing the sequence
    length distribution for each functional group.

    Args:
        enzyme: DataFrame containing only enzyme proteins.
        receptor: DataFrame containing only receptor proteins.
        structural: DataFrame containing only structural proteins.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle("Sequence Length Distribution by Functional Group", fontsize=16)

    sns.histplot(data=enzyme, x='Length', ax=axes[0])
    axes[0].set_title('Enzyme')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Count')

    sns.histplot(data=receptor, x='Length', ax=axes[1])
    axes[1].set_title('Receptor')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Count')

    sns.histplot(data=structural, x='Length', ax=axes[2])
    axes[2].set_title('Structural')
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('length_distribution.png')


def main() -> None:
    """
    Runs the full EDA pipeline: loads data, computes features,
    prints summaries, and generates all visualizations.
    """
    data = load_dataset("protein_dataset.csv")
    data = calculate_biochemical_features(data)
    enzyme, receptor, structural = sort_data_func_groups(data)
    dataframe_shape(data)
    null_data(data)
    individual_summaries(data)
    grouped_summaries(data)
    aa_groups_boxplot(data)
    plot_heatmap(data)
    plot_histograms_func_type(enzyme, receptor, structural)
    size_hist(enzyme, receptor, structural)


if __name__ == "__main__":
    main()
