import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

m = 1500    # 波长（nm）
c = 3e17    # 光速（nm/s）
w = 0.15 * 2 * np.pi * c / m    # B的角频率（rad/s）
a = 0.10 * m    # B的振幅（nm）
t_total = 40 * m / c    # 总的模拟时间（s）


def motion(t):
    return a * np.sin(w * t) + c * t


def excite(x):
    peak = 0.0
    wave = np.array([0.0, 0.125, 0.25, 0.375,
                     0.5, 0.625, 0.75, 0.875])
    for k in range(8):
        x_t = x + wave[k] * m
        x_n = np.abs(x_t / m - np.rint(x_t / m))
        peak += np.exp(- x_n * 1000) * np.sin(wave[k] * 2.0 * np.pi)
    return peak


if __name__ == '__main__':

    m_list = np.zeros(4000000)
    t_list = np.linspace(0, t_total, len(m_list))

    for i in range(len(m_list)):
        y = motion(t_list[i])
        m_list[i] = excite(y)

    n_list = np.arange(len(m_list))
    plt.plot(n_list, m_list)
    plt.show()  # 输出接受信号的时域谱

    f_list = fft(m_list)
    plt.plot(n_list[:100], np.abs(f_list[:100]))
    plt.scatter([40], [np.abs(f_list[40])], c='green')
    plt.scatter([34], [np.abs(f_list[34])], c='green')
    plt.scatter([46], [np.abs(f_list[46])], c='green')
    plt.show()  # 输出接受信号的频域谱

    print(np.abs(f_list[:160]))
