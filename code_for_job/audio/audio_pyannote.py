import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import librosa
import soundfile as sf
import os

# Check if a GPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the pre-trained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

# Move the pipeline to the selected device
pipeline = pipeline.to(device)

# Run diarization on the specified audio file
# diarization = pipeline("300_AUDIO.wav", num_speakers=2)
diarization = pipeline("300_AUDIO.wav", min_speakers=2, max_speakers=3)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

# Load the RTTM file and parse its content
rttm_file = "audio.rttm"

with open(rttm_file, "r") as file:
    rttm_lines = file.readlines()

# Load the original audio
audio_file = "300_AUDIO.wav"
audio, sr = librosa.load(audio_file, sr=None)

output_directory = "diarization_output"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Extract and save audio segments
for line in rttm_lines:
    parts = line.strip().split()
    start_time = float(parts[3])
    end_time = start_time + float(parts[4])
    speaker_label = parts[7]

    # Convert time to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the segment from the original audio
    segment = audio[start_sample:end_sample]

    # Save the segment to a new audio file in WAV format
    output_file = os.path.join(output_directory, f"{speaker_label}_{start_time}_{end_time}.wav")

    sf.write(output_file, segment, sr)

