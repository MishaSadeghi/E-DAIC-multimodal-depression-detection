import subprocess
import torch

checkpoint = torch.load('NISQA-master/weights/nisqa.tar')
checkpoint['args']['ms_max_segments'] = 50000
torch.save(checkpoint, 'NISQA-master/weights/nisqa_2.tar')

# Define the command as a list of arguments
# command = [
#     "python", "NISQA-master/run_predict.py",
#     "--mode", "predict_file",
#     "--pretrained_model", "NISQA-master/weights/nisqa_2.tar",
#     "--deg", "/home/woody/empk/empk004h/DAIC_dataset/audio/321_AUDIO.wav",
#     "--output_dir", "/home/hpc/empk/empk004h/depression-detection/audio_quality_check/results"
# ]

# python run_predict.py --mode predict_file --pretrained_model weights/nisqa.tar --deg /home/woody/empk/empk004h/DAIC_dataset/audio/303_AUDIO.wav --output_dir /home/hpc/empk/empk004h/depression-detection/audio_quality_check/results

# Analyse the whole directory 
command = [
    "python", "NISQA-master/run_predict.py",
    "--mode", "predict_dir",
    "--pretrained_model", "NISQA-master/weights/nisqa_2.tar",
    "--data_dir", "/home/woody/empk/empk004h/DAIC_dataset/audio",
    "--num_workers", "0",
    "--bs", "10",
    "--output_dir", "/home/hpc/empk/empk004h/depression-detection/audio_quality_check/results"
]

# python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /path/to/folder/with/wavs --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results

# Use subprocess.run to execute the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output
print("Standard Output:", result.stdout)
print("Standard Error:", result.stderr)

# Check if the command was successful
if result.returncode == 0:
    print("Command executed successfully.")
else:
    print("Command failed with return code:", result.returncode)

