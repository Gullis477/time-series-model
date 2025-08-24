from emitter_data import Emitter, get_signals, generate_balanced_data
import numpy as np

main_seed = np.random.SeedSequence(42)
data_seed, noise_seed = main_seed.spawn(2)
data_generator = np.random.default_rng(data_seed)
noise_generator = np.random.default_rng(noise_seed)


emitters = generate_balanced_data(total_count=18, rng=data_generator)
signals = get_signals(emitters=emitters, number_of_signals=3, rng=noise_generator)

signals_array = np.array(signals, dtype=object)
data_generator.shuffle(signals_array)
signals = signals_array.tolist()
