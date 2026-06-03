import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset(file: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file: Path to the CSV file.

    Returns:
        A DataFrame containing the loaded data.
    """
    return pd.read_csv(file)

def calculate_biochemical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes biochemical property ratios from raw protein sequences
    and adds them as new columns to the DataFrame.

    Features computed:
        - hydrophobic_ratio: fraction of hydrophobic residues (AVILMFYWC)
        - charged_pos_ratio: fraction of positively charged residues (KRH)
        - charged_neg_ratio: fraction of negatively charged residues (DE)
        - charge_balance: charged_pos_ratio - charged_neg_ratio
        - polar_ratio: fraction of polar uncharged residues (STNQ)
        - aromatic_ratio: fraction of aromatic residues (FYW)

    Args:
        data: The protein DataFrame containing a 'Sequence' column.

    Returns:
        The DataFrame with six new biochemical feature columns added.
    """
    hydrophobic = set("AVILMFYWC")
    charged_pos = set("KRH")
    charged_neg = set("DE")
    polar = set("STNQ")
    aromatic = set("FYW")

    def ratio(seq: str, aa_set: set) -> float:
        """Returns the proportion of amino acids in aa_set within seq."""
        return sum(seq.count(aa) for aa in aa_set) / len(seq) if pd.notna(seq) and len(seq) > 0 else 0

    data["hydrophobic_ratio"] = data["Sequence"].apply(lambda seq: ratio(seq, hydrophobic))
    data["charged_pos_ratio"] = data["Sequence"].apply(lambda seq: ratio(seq, charged_pos))
    data["charged_neg_ratio"] = data["Sequence"].apply(lambda seq: ratio(seq, charged_neg))
    data["charge_balance"] = data["charged_pos_ratio"] - data["charged_neg_ratio"]
    data["polar_ratio"] = data["Sequence"].apply(lambda seq: ratio(seq, polar))
    data["aromatic_ratio"] = data["Sequence"].apply(lambda seq: ratio(seq, aromatic))

    return data

def sort_data_func_groups(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into three separate DataFrames by functional group.

    Args:
        data: The protein DataFrame containing a 'functional_group' column.

    Returns:
        A tuple of three DataFrames: (enzyme, receptor, structural).
    """
    enzyme = data[data["functional_group"] == "Enzyme"]
    receptor = data[data["functional_group"] == "Receptor"]
    structural = data[data["functional_group"] == "Structural"]
    return enzyme, receptor, structural