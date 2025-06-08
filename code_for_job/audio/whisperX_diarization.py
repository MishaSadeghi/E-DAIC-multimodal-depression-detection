import whisperx
import gc 
import requests
import os

# Set the proxy environment variables based on your cluster's requirements
# Replace 'http://proxy:80' with the actual proxy server and port.
os.environ['http_proxy'] = 'http://proxy:80'
os.environ['https_proxy'] = 'http://proxy:80'
# Or, use the uppercase versions if needed
# os.environ['HTTP_PROXY'] = 'http://proxy:80'
# os.environ['HTTPS_PROXY'] = 'http://proxy:80'

device = "cuda" 
audio_file = "307_AUDIO.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

try:
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print('segments: ', result["segments"]) # before alignment
    print('----------------------------------------')

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print('segments after alignment: ', result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

    # add min/max number of speakers if known
    # diarize_segments = diarize_model(audio)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=3)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # print('diarize_segments: ', diarize_segments)
    # print('----------------------------------------')
    print('segments assigned: ', result["segments"]) # segments are now assigned speaker IDs
    print('----------------------------------------')

    # # Print each segment with its detected speaker
    # for segment in result["segments"]:
    #     print(f'Speaker: {segment["speaker"]}, Text: {segment["text"]}')

    for segment in result.get("segments", []):
        if "speaker" in segment:
            print(f'Speaker: {segment["speaker"]}, Text: {segment["text"]}')
        else:
            print(f'Speaker: Unknown, Text: {segment["text"]}')


except requests.exceptions.Timeout:
    print("HTTP/HTTPS timeout occurred. Check your internet connection or the server's availability.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Unset proxy environment variables after executing the code if needed
# del os.environ['http_proxy']
# del os.environ['https_proxy']
# Or, use the uppercase versions if needed
# del os.environ['HTTP_PROXY']
# del os.environ['HTTPS_PROXY']

