#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
DEDUP=remove_duplication.py
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
  "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz"
  "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
  "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-ru.tsv.gz"
  "http://data.statmt.org/wikititles/v2/wikititles-v2.ru-en.tsv.gz"
  "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-ru.langid.tsv.gz"
  "http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.en.gz"
  "http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.en.translatedto.ru.gz"
  "http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.ru.gz"
  "http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.ru.translatedto.en.gz"
  "http://data.statmt.org/wmt20/translation-task/dev.tgz"
  "http://data.statmt.org/wmt20/translation-task/test.tgz"
)
FILES=(
  "paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz"
  "training-parallel-commoncrawl.tgz"
  "news-commentary-v15.en-ru.tsv.gz"
  "wikititles-v2.ru-en.tsv.gz"
  "WikiMatrix.v1.en-ru.langid.tsv.gz"
  "news.en.gz"
  "news.en.translatedto.ru.gz"
  "news.ru.gz"
  "news.ru.translatedto.en.gz"
  "dev.tgz"
  "test.tgz"
)

CORPORA=(
  #    "news"
  "paracrawl-release1.en-ru.zipporah0-dedup-clean"
  "commoncrawl.ru-en"
  "en-ru/UNv1.0.en-ru"
  "1mcorpus/corpus.en_ru.1m"
  "news-commentary-v15.en-ru"
  "WikiMatrix.v1.en-ru.langid"
  "wikititles-v2.ru-en"
  "global_voices"
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
dev=dev/newstest2020

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
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
        elif [ ${file: -4} == ".gz" ]; then
            gunzip $file
        fi
    fi
done

awk -F "\t" 'BEGIN {OFS=FS} {print $1}' news-commentary-v15.en-ru.tsv > news-commentary-v15.en-ru.en
awk -F "\t" 'BEGIN {OFS=FS} {print $2}' news-commentary-v15.en-ru.tsv > news-commentary-v15.en-ru.ru
cd ..

exit

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 24 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "processing back-trans news dataset"
cat $orig/news.ru.translatedto.en | \
            perl $NORM_PUNC en | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 24 -a -l en >> $tmp/train.tags.$lang.tok.en

cat $orig/news.ru | \
            perl $NORM_PUNC ru | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 24 -a -l ru >> $tmp/train.tags.$lang.tok.ru

 Remove Deplications
paste $tmp/train.tags.$lang.tok.en $tmp/train.tags.$lang.tok.ru | python clean_nmt_data.py --directions en-ru --no-zero-len --max-sent-len 1024 --no-bad-utf --max-jaccard-coef-exclusive 0.05 --filter-equality >> $tmp/tmp
awk -F "\t" '{print $1}'  $tmp/tmp > $tmp/train.tags.$lang.clean.tok.en
awk -F "\t" '{print $2}'  $tmp/tmp > $tmp/train.tags.$lang.clean.tok.ru
rm $tmp/tmp

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newstest2020-enru-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/test.$l
    echo ""
done

echo "pre-processing test data 19..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newstest2019-enru-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $tmp/test19.$l
    echo ""
done

for l in $src $tgt; do
cat $orig/global_voices.${l} | \
            perl $NORM_PUNC ${l} | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 24 -a -l ${l} >> $tmp/global_voices.$l
done

for l in $src $tgt; do
cat $orig/reddit_valid.${l} | \
            perl $NORM_PUNC ${l} | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 24 -a -l ${l} >> $tmp/reddit_valid.$l
done

exit

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.clean.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.clean.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.en-ru
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
  for f in train.$L valid.$L test.$L test19.$L reddit_valid.$L global_voices.$L; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/$f >$tmp/bpe.$f
  done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
  cp $tmp/bpe.test.$L $prep/test.$L
  cp $tmp/bpe.test19.$L $prep/test19.$L
  cp $tmp/bpe.test19.$L $prep/test19.$L
  cp $tmp/bpe.reddit_valid.$L $prep/reddit_valid.$L
  cp $tmp/bpe.global_voices.$L $prep/global_voices.$L
done
