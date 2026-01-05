import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, welch

fs = 16000
duration = 5
N = fs * duration
filter_len = 256
mu = 0.6
eps = 1e-8
dt_threshold = 0.6

np.random.seed(0)

x = np.random.randn(N)
v = 0.5 * np.random.randn(N)

true_echo_path = np.exp(-np.linspace(0, 4, filter_len))
true_echo_path /= np.linalg.norm(true_echo_path)

y = lfilter(true_echo_path, 1, x)
d = y + v

w = np.zeros(filter_len)
x_buf = np.zeros(filter_len)

y_hat = np.zeros(N)
e = np.zeros(N)
dt_flag = np.zeros(N)
w_hist = []

for n in range(N):
    x_buf[1:] = x_buf[:-1]
    x_buf[0] = x[n]

    y_hat[n] = np.dot(w, x_buf)
    e[n] = d[n] - y_hat[n]

    Px = np.dot(x_buf, x_buf)
    Pe = e[n] ** 2

    if Pe / (Px + eps) < dt_threshold:
        w += (mu / (Px + eps)) * e[n] * x_buf
        dt_flag[n] = 0
    else:
        dt_flag[n] = 1

    if n % 200 == 0:
        w_hist.append(w.copy())

w_hist = np.array(w_hist)

frame_len = fs // 10
ERLE_t = []

for i in range(0, N - frame_len, frame_len):
    ERLE_t.append(
        10 * np.log10(
            np.mean(y[i:i+frame_len]**2) /
            (np.mean(e[i:i+frame_len]**2) + eps)
        )
    )

ERLE_t = np.array(ERLE_t)

f, P_d = welch(d, fs, nperseg=1024)
_, P_y = welch(y, fs, nperseg=1024)
_, P_e = welch(e, fs, nperseg=1024)

t = np.arange(N) / fs

plt.figure(figsize=(14, 14))

plt.subplot(5, 1, 1)
plt.plot(t, d, label="Mic Signal (echo + speech)")
plt.plot(t, y, label="True Echo", alpha=0.7)
plt.legend()
plt.grid()

plt.subplot(5, 1, 2)
plt.plot(t, y, label="True Echo")
plt.plot(t, y_hat, label="Estimated Echo", alpha=0.7)
plt.legend()
plt.grid()

plt.subplot(5, 1, 3)
plt.plot(t, e, label="Echo-Cancelled Output")
plt.legend()
plt.grid()

plt.subplot(5, 1, 4)
plt.plot(t, dt_flag, label="Double-Talk Flag")
plt.legend()
plt.grid()

plt.subplot(5, 1, 5)
plt.plot(ERLE_t)
plt.xlabel("Frame Index")
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(f, P_d, label="Mic Signal")
plt.semilogy(f, P_y, label="Echo")
plt.semilogy(f, P_e, label="After Cancellation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(true_echo_path, label="True Echo Path")
plt.plot(w, label="Estimated Echo Path")
plt.legend()
plt.grid()
plt.show()

print(f"Final ERLE (avg of last frames): {np.mean(ERLE_t[-10:]):.2f} dB")
plt.savefig("time_domain_results.png", dpi=300)
plt.savefig("psd_results.png", dpi=300)
plt.savefig("echo_path_results.png", dpi=300)