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
