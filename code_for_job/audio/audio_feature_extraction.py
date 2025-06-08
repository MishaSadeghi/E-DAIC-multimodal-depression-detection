import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import librosa
import librosa.display
import parselmouth
from parselmouth.praat import call
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h5py
# ------------------------------------------------------------------------------------------------
# Reading audios and extracting mfcc features from librosa

# Define the directory path
directory_path = '/home/hpc/empk/empk004h/depression-detection/data/audio/'
print('log_mel_spectrogram')
# Initialize a list to store extracted features
features_list = []

# Define a function to calculate Harmonics-to-Noise Ratio (HNR)
def calculate_hnr(audio_path):
    sound = parselmouth.Sound(audio_path)
    harmonicity = sound.to_harmonicity()
    hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove invalid values
    hnr_mean = np.mean(hnr_values)
    hnr_std = np.std(hnr_values)
    return hnr_values, hnr_mean, hnr_std

# This is the function to measure source acoustics using default male parameters.

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

# Define a function to calculate Mel-scale spectrograms and log-spectrograms
def calculate_spectrograms(audio_path):
    # Load the audio using librosa
    audio, sample_rate = librosa.load(audio_path, sr=None)
    
    # Extract MFCC features using Librosa
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    
    # Calculate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    
    # Convert to log scale
    # log_mel_spectrogram = 10 * np.log10(mel_spectrogram)
    log_mel_spectrogram = 10 * np.log10(mel_spectrogram + 1e-10)

    
    return mfccs, log_mel_spectrogram

# Iterate through audio files and extract features
for file_name in os.listdir(directory_path):
    print('file_name: ', file_name)
    if file_name.endswith(".wav"):
        participant_id = int(file_name.split("_")[0])
        print('participant_id: ', participant_id)

        audio_path = os.path.join(directory_path, file_name)
        
        # Calculate Mel-scale spectrograms and log-spectrograms
        mfccs, log_mel_spectrogram = calculate_spectrograms(audio_path)

        # Load the audio using librosa
        audio, sample_rate = librosa.load(audio_path, sr=None)
        
        # Extract MFCC features using Librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        # Load the audio using Parselmouth
        snd = parselmouth.Sound(audio_path)
        
        # Calculate fundamental frequency (F0)
        f0 = snd.to_pitch()
        f0_values = f0.selected_array['frequency']
        
        # Calculate mean and standard deviation of F0
        f0_mean = np.mean(f0_values)
        f0_std = np.std(f0_values)
        
        # Calculate mean of each MFCC coefficient
        mfccs_mean = np.mean(mfccs, axis=1)

        # Calculate formant frequencies (f1-f4) using Parselmouth
        formants = snd.to_formant_burg()
        formant_values = []
        audio_length = snd.get_total_duration()
        for time in np.linspace(0, audio_length, num=100):  # Adjust num as needed
            formant_values.append(formants.get_value_at_time(1, time))  # Replace 1 with formant index
        
        # Calculate mean and standard deviation of formant frequencies
        formant_mean = np.mean(formant_values, axis=0)
        formant_std = np.std(formant_values, axis=0)     
        
        # Calculate audio intensity
        intensity = snd.to_intensity()
        intensity_values = intensity.values
        
        # Calculate mean and standard deviation of audio intensity
        intensity_mean = np.mean(intensity_values)
        intensity_std = np.std(intensity_values)
        
        # Calculate Harmonics-to-Noise Ratio (HNR)
        raw_hnr_values, hnr_mean, hnr_std = calculate_hnr(audio_path)

        # Calculate jitter and shimmer
        localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, \
        localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer = measurePitch(snd, 75, 300, "Hertz")
        
        # Calculate pause-related features
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 300)
        num_pauses = call(point_process, "Get number of points")
        pause_intervals = []
        for i in range(1, num_pauses + 1):
            pause_start = call(point_process, "Get time from index", i)
            pause_end = call(point_process, "Get time from index", i + 1)
            pause_intervals.append(pause_end - pause_start)
            
        pause_time_mean = np.mean(pause_intervals)
        pause_time_std = np.std(pause_intervals)

        # Measure audio length
        audio_length = call(snd, "Get total duration")
        
        features_list.append({
            'id': participant_id,
            'mfccs_mean': mfccs_mean,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'raw_f0_values': f0_values.tolist(),
            'formant_mean': formant_mean.tolist(),
            'formant_std': formant_std.tolist(),
            'raw_formant_values': formant_values,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'raw_intensity_values': intensity_values,
            'hnr_mean': hnr_mean,
            'hnr_std': hnr_std,
            'raw_hnr_values': raw_hnr_values,
            'localJitter': localJitter,
            'localabsoluteJitter': localabsoluteJitter, 
            'rapJitter': rapJitter, 
            'ppq5Jitter': ppq5Jitter, 
            'ddpJitter': ddpJitter, 
            'localShimmer': localShimmer, 
            'localdbShimmer': localdbShimmer, 
            'apq3Shimmer': apq3Shimmer, 
            'aqpq5Shimmer': aqpq5Shimmer, 
            'apq11Shimmer': apq11Shimmer, 
            'ddaShimmer': ddaShimmer,
            'pause_time_mean': pause_time_mean,
            'pause_time_mean': pause_time_std,
            'audio_length': audio_length,
            'mfccs': mfccs.tolist(),
            'log_mel_spectrogram': log_mel_spectrogram.tolist()
        })
    else:
        print('file is not wav')

    print('file_name: ', file_name, ' feature extraction done!')
    

# Create a DataFrame from the features list
features_df = pd.DataFrame(features_list)

# Print the features DataFrame
print(features_df.head())

# Save the features DataFrame to a CSV file
csv_file_path = 'extracted_features_from_audio.csv'
features_df.to_csv(csv_file_path, index=False)
print('Saving to CSV file is done!')

hdf5_file_path = 'extracted_features_from_audio.h5'
features_df.to_hdf(hdf5_file_path, key='features', mode='w')
print('Saving to HDF5 file is done!')
