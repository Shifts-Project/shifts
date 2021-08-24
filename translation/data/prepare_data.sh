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
  "https://storage.yandexcloud.net/yandex-research/shifts/translation/train-data.tar"
  "https://storage.yandexcloud.net/yandex-research/shifts/translation/dev-data.tar"
)
FILES=(
  "train-data.tar"
  "dev-data.tar"
)

CORPORA=(
  "train-data/paracrawl-release1.en-ru.zipporah0-dedup-clean"
  "train-data/commoncrawl.ru-en"
  "train-data/en-ru/UNv1.0.en-ru"
  "train-data/1mcorpus/corpus.en_ru.1m"
  "train-data/news-commentary-v15.en-ru"
  "train-data/WikiMatrix.v1.en-ru.langid"
  "train-data/wikititles-v2.ru-en"
)

OUTDIR=wmt20_en_ru

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

src=en
tgt=ru
lang=en-ru
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

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

awk -F "\t" 'BEGIN {OFS=FS} {print $1}' train-data/news-commentary-v15.en-ru.tsv > train-data/news-commentary-v15.en-ru.en
awk -F "\t" 'BEGIN {OFS=FS} {print $2}' train-data/news-commentary-v15.en-ru.tsv > train-data/news-commentary-v15.en-ru.ru

awk -F "\t" 'BEGIN {OFS=FS} {print $2}' train-data/wikititles-v2.ru-en.tsv > train-data/wikititles-v2.ru-en.en
awk -F "\t" 'BEGIN {OFS=FS} {print $1}' train-data/wikititles-v2.ru-en.tsv > train-data/wikititles-v2.ru-en.ru

awk -F "\t" 'BEGIN {OFS=FS} {print $2}' train-data/WikiMatrix.v1.en-ru.langid.tsv > train-data/WikiMatrix.v1.en-ru.langid.en
awk -F "\t" 'BEGIN {OFS=FS} {print $3}' train-data/WikiMatrix.v1.en-ru.langid.tsv > train-data/WikiMatrix.v1.en-ru.langid.ru

cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
  rm $tmp/train.tags.$lang.tok.$l
  for f in "${CORPORA[@]}"; do
    echo ${f}
    cat $orig/$f.$l |
      perl $NORM_PUNC $l |
      perl $REM_NON_PRINT_CHAR |
      perl $TOKENIZER -threads 24 -a -l $l >> $tmp/train.tags.$lang.tok.$l
  done
done

echo "processing back-trans news dataset"
cat $orig/train-data/news.ru.translatedto.en |
  perl $NORM_PUNC en |
  perl $REM_NON_PRINT_CHAR |
  perl $TOKENIZER -threads 24 -a -l en >> $tmp/train.tags.$lang.tok.en

cat $orig/train-data/news.ru |
  perl $NORM_PUNC ru |
  perl $REM_NON_PRINT_CHAR |
  perl $TOKENIZER -threads 24 -a -l ru >> $tmp/train.tags.$lang.tok.ru



echo "Remove Deplications, clean data"
paste $tmp/train.tags.$lang.tok.en $tmp/train.tags.$lang.tok.ru | python $CLEAN_DATA --directions en-ru --no-zero-len --max-sent-len 1024 --no-bad-utf --max-jaccard-coef-exclusive 0.05 --filter-equality >>$tmp/tmp
awk -F "\t" '{print $1}' $tmp/tmp > $tmp/train.tags.$lang.clean.tok.en
awk -F "\t" '{print $2}' $tmp/tmp > $tmp/train.tags.$lang.clean.tok.ru
rm $tmp/tmp

echo "pre-processing dev data newstest19..."
for l in $src $tgt; do
  if [ "$l" == "$src" ]; then
    t="src"
  else
    t="ref"
  fi
  grep '<seg id' $orig/dev-data/newstest2019-enru-$t.$l.sgm |
    sed -e 's/<seg id="[0-9]*">\s*//g' |
    sed -e 's/\s*<\/seg>\s*//g' |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/test19.$l
  echo ""
done

echo "pre-processing dev data reddit..."
for l in $src $tgt; do
  cat $orig/dev-data/reddit_dev.${l} |
    perl $NORM_PUNC ${l} |
    perl $REM_NON_PRINT_CHAR |
    perl $TOKENIZER -threads 24 -a -l ${l} >>$tmp/reddit_dev.$l
done

echo "splitting train and valid..."
for l in $src $tgt; do
  awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.clean.tok.$l >$tmp/valid.$l
  awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.clean.tok.$l >$tmp/train.$l
done

TRAIN=$tmp/train.en-ru
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
  cat $tmp/train.$l >>$TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS <$TRAIN >$BPE_CODE

for L in $src $tgt; do
  for f in train.$L valid.$L test19.$L reddit_dev.$L; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/$f >$tmp/bpe.$f
  done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
  cp $tmp/bpe.valid.$L $prep/valid.$L
  cp $tmp/bpe.test19.$L $prep/test19.$L
  cp $tmp/bpe.reddit_dev.$L $prep/reddit_dev.$L
done
