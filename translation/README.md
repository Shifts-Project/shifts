# Machine Translation Track 

Welcome to the Shifts Challenge Translation track!

### Downloading and Processing the data
To download the training and development data run the preprocess script:

```
./data/preprocess.sh
```


### Setting up the baselines

First clone a Fairseq fork which contains...

```
clone MY COOL Set
```

Next, download the baselines models

```
wget https://storage.yandexcloud.net/yandex-research/shifts/translation/baseline-models.tar
tar -xf baseline-models.tar
```

### Running the baselines

```
python3 ~/fairseq-py/generate.py wmt20_en_ru_eval/ --path baseline-models/model1.pt:baseline-models/model2.pt:baseline-models/model3.pt  \\
--max-tokens 4096 --remove-bpe --nbest 5 --gen-subset test >&

python3 ~/fairseq-py/generate.py wmt20_en_ru_eval/ --path ens_enru/model1.pt:ens_enru/model2.pt:ens_enru/model3.pt  --max-tokens 1024 --remove-bpe --nbest 5 --gen-subset test --compute-uncertainty >& ensemble/results-test.txt
for j in $(seq 1 2); do 
    python3 ~/fairseq-py/generate.py wmt20_en_ru_eval/ --path ens_enru/model1.pt:ens_enru/model2.pt:ens_enru/model3.pt --max-tokens 1024 --remove-bpe --nbest 5 --gen-subset test${j} --compute-uncertainty >& ensemble/results-test${j}.txt
done
```

This produces the raw output of the translations and associated uncertainty scores which then need to be processed further.


### Processing the results locally

### Creating a submission
й