import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = os.listdir('data/hms/train_spectrograms/')
print(f"Found {len(files)} spectrogram files.")

print("Read file: ", files[0])

# Read a spectrogram stored in a Parquet file
df = pd.read_parquet(f'data/hms/train_spectrograms/{files[0]}')

# Log transform
spec = np.log1p(df)

# Each column is a frequency bin, each row is a time step
spec = spec.values.astype(np.float32)  # shape: (time, freq)
# Normalize to [0, 1]
spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
# Trasform into image shape like data (1, H, W)
img = spec[None, :, :]

# show the spectrogram
plt.imshow(img[0], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Normalized Amplitude')
plt.title('Spectrogram from Parquet File')
plt.xlabel('Frequency Bin')
plt.ylabel('Time Step')
plt.show()
