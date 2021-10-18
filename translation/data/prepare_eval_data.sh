#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

echo 'Cloning Shifts Challenge Repo '
git clone https://github.com/yandex-research/shifts.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
DEDUP=remove_duplication.py
CLEAN_DATA=shifts/translation/data/clean_nmt_data.py
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
  "https://storage.yandexcloud.net/yandex-research/shifts/translation/eval-data.tar"
)
FILES=(
  "eval-data.tar"
)

CORPORA=(
  "global_voices_eval"
)

OUTDIR=wmt20_en_ru

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
  exit
fi

src=en
tgt=ru
lang=en-ru
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2019
BPE_CODE=$prep/code

cd $orig

for ((i = 0; i < ${#URLS[@]}; ++i)); do
  file=${FILES[i]}
  echo $file
  if [ -f $file ]; then
    echo "$file already exists, skipping download"
  else
    url=${URLS[i]}
    wget "$url"
    if [ -f $file ]; then
      echo "$url successfully downloaded."
    else
      echo "$url not successfully downloaded."
      exit -1
    fi

    if [ ${file: -4} == ".tgz" ]; then
      tar zxvf $file
    elif [ ${file: -4} == ".tar" ]; then
      tar xvf $file
    elif [ ${file: -3} == ".gz" ]; then
      gunzip $file
    fi
  fi
done

cd ..

for l in $src $tgt; do
  for corp in "${CORPORA[@]}"; do
    cat $orig/eval-data/${corp}.${l} |
      perl $NORM_PUNC ${l} |
      perl $REM_NON_PRINT_CHAR |
      perl $TOKENIZER -threads 24 -a -l ${l} >$tmp/${corp}.$l
  done
done



for L in $src $tgt; do
  for corp in "${CORPORA[@]}"; do
    for f in ${corp}.$L; do
      echo "apply_bpe.py to ${f}..."
      python $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/$f >$tmp/bpe.$f
    done
  done
done

for L in $src $tgt; do
  for corp in "${CORPORA[@]}"; do
    cp $tmp/bpe.${corp}.$L $prep/${corp}.$L
  done
done
