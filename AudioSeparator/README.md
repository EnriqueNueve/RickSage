# Cocktails with Robots

## Run src/train.py
```
python3 train.py --gpu 0 --epochs 1 --batch_size 1 --model_type RHA --network_n_sources 2 --network_num_filters_in_encoder 64 --network_num_head_per_att 8 --network_dim_key_att 1024 --network_num_tran_blocks 1 --network_num_chop_blocks 1 --network_chunk_size 256 --data_set /data/sample_data --max_input_length_in_seconds 5 --samplerate_hz 8000 --lr 1e-4
```

## Done 
* Made SepFormer 
* Made LinFormer Transformer Block 
* Made Linear Attention with Kernel Trick 
* Made Linear Attention with Kernel Trick with Transformer 
* Made SineSPE work with GetQKV
* merge SineSPE/SPEFilter/LinearTransformer

## To Do
1. Implement fuss metric tools: https://github.com/google-research/sound-separation/blob/master/models/dcase2020_fuss_baseline/evaluate_lib.py


## To Do not priority 
* Make train implement of SepFormer 
* Make RHA without SPE and just sinusodial 
* Implment ConvSPE
* Make end to end formula sheet 

## Data sets: https://paperswithcode.com/task/audio-source-separation
#### Fuss
* https://github.com/google-research/sound-separation/blob/master/models/dcase2020_fuss_baseline/evaluate_lib.py
* https://zenodo.org/record/3694384#.YPsbaRNKgq1
* FUSS_ssdata.tar.gz
#### LibriMix 
* https://github.com/JorisCos/LibriMix
#### WHAM!
* https://wham.whisper.ai/
* wget -c https://storage.googleapis.com/whisper-public/whamnoise.zip
#### MUSDB18
* https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
#### AVSpeech
* https://looking-to-listen.github.io/avspeech/



## Publication venues 
* https://2022.ieeeicassp.org/ , deadline October 01, 2021

## Scratch thoughts 
* Think of procedure for auto-setting dimensions (padding, constant scaling factors, etc)
* Plan out file directory scheme before assembling intial training model protocol (parse arguments, modularity, data set, sample rate, max len of seq, dynamic mixing).

## Paper
* If performance isn't as good, show if its faster

## Resources
* https://source-separation.github.io/tutorial/landing.html
* https://github.com/speechbrain/speechbrain/
* https://github.com/JusperLee/Speech-Separation-Paper-Tutorial
* https://github.com/iver56/audiomentations
* https://github.com/sp-uhh/dual-path-rnn/blob/master/src/network.py
