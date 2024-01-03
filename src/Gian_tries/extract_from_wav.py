import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_audio_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract MFCCs, a set of coefficients representing the short-term power spectrum of a sound signal
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Extract Chroma Feature, represents the 12 different pitch classes as features, capturing tonal content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Extract Spectral Contrast, measures the difference in amplitude between peaks and valleys in the spectrum
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Extract Spectral Centroid, indicates the center of mass of the spectrum
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Extract Spectral Bandwidth, measures the width of the spectrum
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    print('MFCCs:', mfccs.shape)
    print('Chroma:', chroma.shape)
    print('Contrast:', contrast.shape)
    print('Centroid:', centroid.shape)
    print('Bandwidth:', bandwidth.shape)

    print('MFCCs:', mfccs)
    print('\nChroma:', chroma)
    print('\nContrast:', contrast)
    print('\nCentroid:', centroid)
    print('\nBandwidth:', bandwidth)

    return mfccs, chroma, contrast, centroid, bandwidth

def plot_features(features, feature_names):
    plt.figure(figsize=(15, 10))

    for i in range(len(features)):
        plt.subplot(5, 1, i+1)
        librosa.display.specshow(features[i], x_axis='time')
        plt.colorbar()
        plt.title(feature_names[i])

    plt.tight_layout()
    plt.show()

# Example usage
file_path = '/media/gian/Common Storage/SDU Lessons/Deep Neural Netwoks/Project/Identity-recognition-from-audio/data/audio_dataset/0_both/her_ga_r_det_godt_01_10_21/segment0005.wav'
mfccs, chroma, contrast, centroid, bandwidth = extract_audio_features(file_path)

# Feature names for plotting
feature_names = ['MFCCs', 'Chroma Feature', 'Spectral Contrast', 'Spectral Centroid', 'Spectral Bandwidth']

# Plot the features
plot_features([mfccs, chroma, contrast, centroid, bandwidth], feature_names)
