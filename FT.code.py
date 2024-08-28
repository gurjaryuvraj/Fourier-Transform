import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('seaborn-poster')
# %matplotlib inline
# # sampling rate
# sr = 100000
# # sampling interval
# ts = 1.0/sr
sr = 100
#sampling interval
si = 1/sr
t = np.arange(0,1,si)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)
# 3 sin (2 pi t)

freq = 4
x += np.sin(2*np.pi*freq*t)
# sin (2 pi t)
freq = 7
x += 0.5* np.sin(2*np.pi*freq*t)
# 0.5 sin( 2 pi t)
plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')
plt.show()
print(len(x))


def FT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return X

X = FT(x)

# calculate the frequency
N = len(X)
print(N)
n = np.arange(N)
T = N/sr
freq = n/T
# print(X)
plt.figure(figsize = (8, 6))
# plt.stem(freq, abs(X), 'b', \
#          markerfmt=" ", basefmt="-b")
plt.stem(freq, abs(X), 'b',markerfmt="" )
plt.xlabel('Freq (Hz)')
plt.ylabel('FT Amplitude |X(freq)|')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(freq, abs(X), color='blue')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum of x')
plt.grid(True)  # Add grid lines for better readability
plt.show()

def gen_sig(sr):
    '''
    function to generate
    a simple 1D signal with
    different sampling rate
    '''
    ts = 1.0/sr
    t = np.arange(0,1,ts)

    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)
    return x

# sampling rate =2000
sr = 2000
%timeit FT(gen_sig(sr))

# sampling rate 20000
sr = 20000
%timeit FT(gen_sig(sr))
