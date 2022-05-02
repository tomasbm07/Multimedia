"""
# *Exercicio 2.3*

> References: <br>
    - https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature <br>
    - https://en.wikipedia.org/wiki/Mel_scale <br>
    - https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html <br>
    - https://gist.github.com/bmcfee/746e572232be36f3bd462749fb1796da <br>
    - https://github.com/librosa/librosa <br>
    - https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py <br>
    - https://www.researchgate.net/publication/220723537_Finding_An_Optimal_Segmentation_for_Audio_Genre_Classification#pf2 <br>
    - https://en.wikipedia.org/wiki/Octave_band <br>
    - https://en.wikipedia.org/wiki/Spectral_flatness <br>
    - https://en.wikipedia.org/wiki/Root_mean_square <br>
    - https://github.com/scipy/scipy <br>
    - http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

"""
import numpy as np
from scipy import fft as scf
from scipy import signal as scsg
from scipy import stats as scs

def hz2mel(f):
    return 2595 * np.log10(1 + f/700)

def mel2hz(m):
    return 700*(10**(m/2595) - 1)

def get_filterbank(sr, n_fft, frame_length, n=40, fmin = 20, fmax=sr/2):
    mel_min = hz2mel(fmin)
    mel_max = hz2mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n+2)
    
    hz_points = mel2hz(mel_points)
    hz_bins = np.floor( (n_fft + 1)/sr * hz_points).astype(int)
    
    fbank = np.zeros( (n, frame_length) )
    for m in range(1, n + 1):
        fleft = hz_bins[m-1]
        fcenter = hz_bins[m]
        fright = hz_bins[m+1]
        
        for k in range(fleft, fcenter+1):
            fbank[m - 1, k] = (k - hz_bins[m - 1])/(hz_bins[m] - hz_bins[m - 1])
        for k in range(fcenter, fright+1):
            fbank[m - 1, k] = (hz_bins[m + 1] - k )/(hz_bins[m + 1] - hz_bins[m])
    
    return fbank

def mfcc(pow_spec, sr, n_fft, hop_length, mel_filter, n_mfcc = 13, lifter=0):
    mel_spec = np.dot( mel_filter, pow_spec)
    mel_spec = 20 * np.log10(mel_spec)
    
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)  # Numerical Stability
    
    mel_spec = scf.dct(mel_spec, norm='ortho', n=n_mfcc)
    return mel_spec

def spectral_centroid(mag, freqs):
    return np.sum(mag*freqs)/np.sum(mag)

def min_max_scale(y):
    min_v = y.min()
    max_v = y.max()
    return (y-min_v)/(max_v-min_v)

def spectral_bandwith(centroid, S, freq, p=2):
    #based in librosa
    deviation = np.abs(np.subtract.outer(centroid[0], freq).T)
    S = np.apply_along_axis(min_max_scale, 0, S)
    return np.sum(S * deviation ** p, axis = -2, keepdims=True) ** (1.0 / p)

def spectral_contrast(mag, sr, freq, n_bands = 6):
    low_f = 20
    
    octa = np.zeros(n_bands + 2)
    octa[1:] = low_f * (2.0 ** np.arange(0, n_bands + 1))
    
    peaks = np.zeros_like(mag)
    valleys = np.zeros_like(mag) 
    
    for i in range(len(octa) - 1):
        index = np.where((freq > octa[i]) & (freq < octa[i+1]))[0]
        freq_band = freq[ index ]
        mag_band = mag[ index ] 
        #print(freq_band)

def spectral_flateness(pow_mag):
    numerator = np.exp( np.log(pow_mag).mean() )
    denominator = pow_mag.mean()
    return numerator/denominator

def spectral_rollof(mag, freq, cutoff_perc):
    cum_mag_sum = np.cumsum(mag)
    threshold = cum_mag_sum[-1] * cutoff_perc
    selected = np.where(cum_mag_sum < threshold)[0]
    return freq[selected].max()

def root_mean_square(power_spec):
    power_spec[0]/=2
    return np.sqrt( power_spec.sum()/len(power_spec)**2 )

def zero_crossing_rate(y):
    pos = (y>0)
    neg = (y<0)

    down_cross = np.logical_and(pos[:-1], neg[1:]).sum()
    up_cross = np.logical_and(neg[:-1], pos[1:]).sum()

    return (down_cross + up_cross)/len(y)

def get_time_frames(y, nperseg, noverlap):
    step    = nperseg - noverlap
    shape   = y.shape[:-1] + ((y.shape[-1]-noverlap) // step, nperseg)
    strides = y.strides[:-1] + (step * y.strides[-1], y.strides[-1])
    result  = np.lib.stride_tricks.as_strided(y, shape=shape,
                                             strides=strides, writeable=True)
    result  = result.swapaxes(0,1)
    return result

def features(y, 
             sr=22050, 
             hop_length = 512, 
             n_mfcc = 13, 
             n_fft=2048
):
    f, t, mag = scsg.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, window="hann", padded=False) 
    mag *= (n_fft/2) # ratio with librosa result is 1024
    
    mag = np.abs(mag)
    power = mag**2
    
    #TODO Refazer calculo de mfcc
    mel_filter = get_filterbank(sr, n_fft, len(f), n=128)
    mfccs = np.apply_along_axis( mfcc, 0, power, sr, n_fft, hop_length, mel_filter)
    
    spec_centroid = np.apply_along_axis(spectral_centroid, 0, mag, f).reshape(1,mag.shape[1]) 
    
    #valores não são iguas, possivelmente devido à normalização
    spec_bdwidth = spectral_bandwith(spec_centroid, mag, f)
    
    #TODO LATER(nao e preciso)
    #spec_cont = np.apply_along_axis(spectral_contrast, 0 , y , sr, f)
    spec_flat=np.apply_along_axis(spectral_flateness, 0, power).reshape(1,mag.shape[1]) 
    
    spec_rollof = np.apply_along_axis(spectral_rollof, 0, mag, f, 0.85).reshape(1,mag.shape[1]) 
    
    rms = np.apply_along_axis(root_mean_square, 0, power).reshape(1,mag.shape[1])
    
    time_frames = get_time_frames(y, n_fft, n_fft - hop_length)
    zcr = np.apply_along_axis(zero_crossing_rate, 0, time_frames).reshape(1,1291) 

    return (mfccs,
            spec_centroid, 
            spec_bdwidth, 
            spec_flat, 
            spec_rollof, 
            rms,
            zcr
           )

mfccs, cent, bdwd, flat, roll, rms, zcr=features(y)