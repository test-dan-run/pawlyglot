import librosa
import librosa.filters
import numpy as np
from scipy import signal

def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _normalize(S, clip_norm=True, symmetric_mels=True, max_abs_value=4, min_level_db=-100):
    if clip_norm:
        if symmetric_mels:
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                           -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))

def melspectrogram(
        wav, n_fft=800, hop_size=200, window_size=800, 
        sample_rate=16000, num_mels=80, fmin=55, fmax=7600, preemphasis=0.97,
        min_level_db=-100,  ref_level_db=20, max_abs_value=4,
        signal_norm=True, clip_norm=True, symmetric_mels=True
        ):

    wav = signal.lfilter([1, -preemphasis], [1], wav)
    spect = librosa.stft(
        y=wav, n_fft=n_fft, hop_length=hop_size, win_length=window_size)
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    mels = _amp_to_db(np.dot(mel_basis, np.abs(spect)), min_level_db=min_level_db)
    mels = mels - ref_level_db

    if signal_norm:
        mels = _normalize(
            mels, 
            clip_norm=clip_norm, symmetric_mels=symmetric_mels, 
            max_abs_value=max_abs_value, min_level_db=min_level_db)

    return mels
