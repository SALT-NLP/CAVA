# Speaker Diarizatin Data Collection Pipeline

The pipeline runs in 2 steps

## Download AMI meetings wav files from AMI corpus(Files are large, recommend to do it on background)

```sh
sh download_ami.sh
```

This download all wav files and save to `data/SpeakerDiarization/amicorpus`

## Get Speaker Diarization Dataset
To prepare the dataset for Speaker Diarization task(in which you give the model an audio and the corresponding transcript without speaker label, and prompt the model to assign each sentence with a speaker), run `dataset_prepare.py`. The `sentences` parameter sets the number of sentences you want to have in the audio and transcript(default to 20).

```sh
python dataset_prepare.py
```

The dataset config will be saved in `data/SpeakerDiarization/audio_inputs.jsonl`

Now you can safely delete `data/SpeakerDiarization/amicorpus`

## Alternative: You can directly download and process the data from huggingface
Please cd back to the home directory of CATS first, then run this script:
```sh
python data/convert_from_hf.py --dataset rma9248/CATS-ami-speaker-diarization-audio --split train --audio-dir SpeakerDiarization --output audio_inputs.jsonl --preserve-columns --limit 1000
```


