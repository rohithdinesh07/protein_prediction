"""
load_data.py
Rohith Dinesh
CSE 163

Downloads raw protein data from the UniProt REST API for three functional
groups (enzymes, receptors, and structural proteins) and saves the combined
dataset as a CSV file. No cleaning or feature calculations are performed.

"""

import requests
import pandas as pd
import time
from io import StringIO


UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

FIELDS = ",".join([
    "accession",
    "protein_name",
    "sequence",
    "length",
    "mass",
    "cc_function",
    "keyword",
])

FUNCTIONAL_GROUPS = {
    "Enzyme":      "keyword:KW-0418 AND reviewed:true",
    "Receptor":    "keyword:KW-0675 AND reviewed:true",
    "Structural":  "keyword:KW-0261 AND reviewed:true",
}

SAMPLE_SIZE = 500


def fetch_group(label: str, query: str, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Queries the UniProt REST API for a single functional group and
    returns the results as a DataFrame with a functional_group label column.

    Args:
        label: The name of the functional group (e.g. 'Enzyme').
        query: The UniProt query string used to filter proteins.
        size: The number of proteins to retrieve. Defaults to SAMPLE_SIZE.

    Returns:
        A DataFrame containing the retrieved proteins with a
        'functional_group' column added.
    """
    print(f"Fetching {label}s ...", end=" ", flush=True)
    params = {
        "query":  query,
        "fields": FIELDS,
        "format": "tsv",
        "size":   size,
    }
    resp = requests.get(UNIPROT_API, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep="\t")
    df["functional_group"] = label
    print(f"{len(df)} rows")
    return df


def main() -> None:
    """
    Fetches protein data for all functional groups, combines them into
    a single DataFrame, and saves the result to protein_dataset.csv.
    """
    frames = []
    for label, query in FUNCTIONAL_GROUPS.items():
        df = fetch_group(label, query)
        frames.append(df)
        time.sleep(1)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv("protein_dataset.csv", index=False)
    print(f"\nSaved -> protein_dataset.csv")
    print(f"Shape: {combined.shape[0]} rows x {combined.shape[1]} columns")


if __name__ == "__main__":
    main()
