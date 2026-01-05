import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ============================================================
# 1. Load audio files
# ============================================================

fs1, clean1 = wavfile.read("clean1.wav")
fs2, clean2 = wavfile.read("clean2.wav")
fs, x = wavfile.read("mixed.wav")

assert fs == fs1 == fs2

clean1 = clean1.astype(np.float64)
clean2 = clean2.astype(np.float64)
x = x.astype(np.float64)

if clean1.ndim > 1:
    clean1 = clean1[:,0]
if clean2.ndim > 1:
    clean2 = clean2[:,0]
if x.ndim > 1:
    x = x[:,0]

N = min(len(clean1), len(clean2), len(x))
clean1 = clean1[:N]
clean2 = clean2[:N]
x = x[:N]

t = np.arange(N) / fs

# ============================================================
# 2. FFT
# ============================================================

X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, 1/fs)
mag = np.abs(X)

# ============================================================
# 3. Robust thresholding
# ============================================================

median = np.median(mag)
mad = np.median(np.abs(mag - median))
threshold = median + 6 * mad

active = mag > threshold

# ============================================================
# 4. Band detection with gap merging
# ============================================================

idx = np.where(active)[0]

MAX_GAP = int(30 * N / fs)

bands = []
current = [idx[0]]

for i in idx[1:]:
    if i - current[-1] <= MAX_GAP:
        current.append(i)
    else:
        bands.append(np.array(current))
        current = [i]

bands.append(np.array(current))

bands = [b for b in bands if np.mean(freqs[b]) > 0]

energies = [np.sum(mag[b]**2) for b in bands]
top2 = np.argsort(energies)[-2:]
band1, band2 = [bands[i] for i in top2]

# ============================================================
# 5. Build binary masks
# ============================================================

mask1 = np.zeros_like(X)
mask2 = np.zeros_like(X)

mask1[band1] = 1
mask2[band2] = 1

mask1[-band1] = 1
mask2[-band2] = 1

# ============================================================
# 6. Apply masks (frequency domain only)
# ============================================================

X1 = X * mask1
X2 = X * mask2

# ============================================================
# 7. Plot spectra ONLY
# ============================================================

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(freqs[freqs > 0], mag[freqs > 0])
plt.axhline(threshold, color='r', linestyle='--')
plt.title("Mixed Signal Spectrum")
plt.ylabel("|X(f)|")

plt.subplot(3,1,2)
plt.plot(freqs[freqs > 0], np.abs(X1)[freqs > 0])
plt.title("Recovered Spectrum – Source 1")
plt.ylabel("|X₁(f)|")

plt.subplot(3,1,3)
plt.plot(freqs[freqs > 0], np.abs(X2)[freqs > 0])
plt.title("Recovered Spectrum – Source 2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X₂(f)|")

plt.tight_layout()
plt.show()