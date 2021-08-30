# Machine Translation Track 

Welcome to the Shifts Challenge Translation track! In this readme we will provide an example of how to setup and run our baseline model, which is based on the paper [Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/pdf?id=jN5y-zb5Q7m) from ICLR2021. 

### Downloading and Processing the data
To download the training and development data run the preprocess script:

```
chmod +x ./shifts/translation/data/prepare_data.sh 
./shifts/translation/data/prepare_data.sh 
```


### Setting up the baselines

First clone a Fairseq fork which contains an implementation of [Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/pdf?id=jN5y-zb5Q7m). Note that code is a little outdated and uses Fairseq 0.7. We plan to create a cleaner up-to-date implementation soon. 

```
git clone https://github.com/KaosEngineer/structured-uncertainty.git
cd structured-uncertainty 
python3 -m pip install --user --no-deps --editable .
```

Download the baselines models

```
wget https://storage.yandexcloud.net/yandex-research/shifts/translation/baseline-models.tar
tar -xf baseline-models.tar
```

### Pre-process the data into Fairseq format

If you are pre-processing your own data, use the following command:
```
python3 structured-uncertainty/fairseq-preprocess.py  --source-lang en --target-lang ru \\
--trainpref wmt20_en_ru/train --validpref wmt20_en_ru/valid --testpref wmt20_en_ru/test19,wmt20_en_ru/reddit_dev  \\
--destdir data-bin/wmt20_en_ru --thresholdtgt 0 --thresholdsrc 0  --workers 24
```

If you are using the provided baseline models, please pre-process using the following command:
```
python3 structured-uncertainty/fairseq-preprocess.py  --srcdict baseline-models/dict.en.txt --tgtdict baseline-models/dict.ru.txt --source-lang en --target-lang ru \\
--trainpref wmt20_en_ru/train --validpref wmt20_en_ru/valid --testpref wmt20_en_ru/test19,wmt20_en_ru/reddit_dev  \\
--destdir data-bin/wmt20_en_ru --thresholdtgt 0 --thresholdsrc 0  --workers 24
```

### Training a model

If you don't want to use the downloaded baseline and instead want to train your model, you can run the following command:

```
python3 structured-uncertainty/train.py data-bin/wmt20_en_ru --arch transformer_wmt_en_de_big --share-decoder-input-output-embed --fp16 --memory-efficient-fp16 --num-workers 16 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 5120 --save-dir MODEL_DIR --max-update 50000 --update-freq 16 --keep-last-epochs 10 --seed 0
```

This was used to produce the baselines, with only the seed varying.

### Running the baselines

Run single model baseline:
```
mkdir single 
for i in test test1; do 
    python3 structured-uncertainty//generate.py wmt20_en_ru/ --path baseline-models/model1.pt --max-tokens 4096 --remove-bpe --nbest 5 --gen-subset ${i} >& single/results_${i}.txt
done
```

Run ensemble baseline:
```
mkdir ensemble
for i in test test1; do 
    python3 structured-uncertainty/generate.py wmt20_en_ru/ --path baseline-models/model1.pt:baseline-models/model2.pt:baseline-models/model3.pt  --max-tokens 1024 --remove-bpe --nbest 5 --gen-subset ${i} --compute-uncertainty >& ensemble/results-${i}.txt
done
```

This produces the raw output of the translations and associated uncertainty scores which then need to be processed further.


### Processing the results locally and create a submission

```
chmod +x ./shifts/translation/assessment/eval_single.sh
chmod +x ./shifts/translation/assessment/eval_ensemble.sh

./shifts/translation/assessment/eval_single.sh
./shifts/translation/assessment/eval_ensemble.sh
```
