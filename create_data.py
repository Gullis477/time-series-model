from typing import List, Tuple
from emitter_data import Emitter, get_signals, generate_balanced_data  # type: ignore
import numpy as np
import pandas as pd


def create_data(
    seed: int, number_of_emitters: int, number_of_signals: int
) -> List[Tuple[pd.DataFrame, int]]:
    """
    Generates a list of shuffled signal data from a specified number of emitters.

    This function creates a balanced set of emitters and generates signal data for each.
    The total number of emitters must be divisible by 9 to ensure a balanced dataset
    across different emitter types.

    Args:
        seed (int): The random seed to ensure reproducibility of the generated data.
        number_of_emitters (int): The total number of emitters to generate.
                                  Must be divisible by 9.
        number_of_signals (int): The number of signals to generate for each emitter.

    Returns:
        List[Tuple[pd.DataFrame, int]]: A shuffled list of tuples. Each tuple contains
                                        a pandas DataFrame with the signal data and an
                                        integer representing the corresponding label.
    """
    if number_of_emitters % 9 != 0:
        raise ValueError("The number of emitters must be divisible by 9.")

    main_seed = np.random.SeedSequence(seed)
    data_seed, noise_seed = main_seed.spawn(2)
    data_generator = np.random.default_rng(data_seed)
    noise_generator = np.random.default_rng(noise_seed)

    emitters = generate_balanced_data(
        total_count=number_of_emitters, rng=data_generator
    )
    signals = get_signals(
        emitters=emitters, number_of_signals=number_of_signals, rng=noise_generator
    )

    signals_array = np.array(signals, dtype=object)
    data_generator.shuffle(signals_array)
    signals = signals_array.tolist()

    return signals


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales specific columns in a DataFrame to the [-1, 1] range
    using predetermined min and max values.

    Args:
        df (pd.DataFrame): A DataFrame containing the 'freq', 'pw', 'bw',
                        and 'pri' columns.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns scaled.
                    Values outside the original range will result in
                    scaled values outside of [-1, 1].
    """

    freq_min, freq_max = 3.5e9, 18.5e9
    pri_min, pri_max = 10e-6, 600e-6
    pw_min, pw_max = 2e-7, 1e-4
    bw_min, bw_max = 1 / pw_max, 1000 / pri_min

    df_scaled = df.copy()

    def min_max_scale(series, min_val, max_val):
        return (series - min_val) / (max_val - min_val) * 2 - 1

    df_scaled["freq"] = min_max_scale(df_scaled["freq"], freq_min, freq_max)
    df_scaled["pw"] = min_max_scale(df_scaled["pw"], pw_min, pw_max)
    df_scaled["bw"] = min_max_scale(df_scaled["bw"], bw_min, bw_max)
    df_scaled["pri"] = min_max_scale(df_scaled["pri"], pri_min, pri_max)

    return df_scaled


if __name__ == "__main__":
    signals = create_data(42, 9, 1)
    test_df = signals[0][0]

    scaled_df = scale_data(test_df)
    print(scaled_df.describe())
