from pathlib import Path

import pandas as pd


VALID_ANIMALS = {"human", "pigeon", "rat"}


def read_data(animal="human", folder="../../Data/clean_data", test=False):
    """
    Read cleaned behavioral data for one animal group.

    Parameters
    ----------
    animal : {"human", "pigeon", "rat"}
        Which animal dataset to load.

    folder : str or pathlib.Path
        Folder containing the cleaned CSV files.

    test : bool
        If True, load the smaller test dataset.

    Returns
    -------
    pd.DataFrame
        Behavioral data with an added boolean `accuracy` column.
    """
    if animal not in VALID_ANIMALS:
        raise ValueError(
            f"animal must be one of {sorted(VALID_ANIMALS)}, got {animal!r}"
        )

    folder = Path(folder)

    prefix = "test_" if test else ""
    file_path = folder / f"{prefix}{animal}_data.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find data file: {file_path}")

    print(f"Reading data for animal: {animal}")
    data = pd.read_csv(file_path)

    data["accuracy"] = data["resp"] == data["truth"]

    return data


def get_Xf(data):
    X=(data[['stim.Orientation','stim.Frequency']].values)/100
    f = (data['truth']-1).values
    return X,f
    
def get_resp(data):
    return (data['resp']-1).values