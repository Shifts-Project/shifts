#!/bin/bash

output_dir=single

# Get hypotheses and eval

(
  cd ${output_dir}
  #Obtain all measures of uncertainty
  for i in test test1; do
    rm -rf ${i}
    mkdir ${i}

    #grep "^H-[0-9]*\s" results-${i}.txt  > ${i}/hypos_tmp.txt
    grep "^H-[0-9]*\s" results-${i}.txt | sed -e "s/^H\-//" >${i}/hypos_tmp.txt
    grep "^T-[0-9]*\s" results-${i}.txt | sed -e "s/^T\-//" >${i}/refs_tmp.txt

    awk -F '\t' '{print $1}' ${i}/hypos_tmp.txt >${i}/hypo_ids.txt
    awk -F '\t' '{print $2}' ${i}/hypos_tmp.txt >${i}/hypo_likelihoods.txt
    awk -F '\t' '{print $3}' ${i}/hypos_tmp.txt >${i}/hypos.txt

    awk -F '\t' '{print $1}' ${i}/refs_tmp.txt >${i}/ref_ids.txt
    awk -F '\t' '{print $2}' ${i}/refs_tmp.txt >${i}/refs.txt

    # Clean up
    rm ${i}/hypos_tmp.txt ${i}/refs_tmp.txt
  done
)

#Evaluate submission
python shifts/translation/assessment/evaluate_dev.py single/test single/test1 --save_path ./results-single.txt --beam_width 5 --nbest 5

# Prepare submission
python shifts/translation/assessment/create_submission.py single/test single/test1 --save_path ./submission-single.json --beam_width 5 --nbest 5 --uncertainty_metric NLL
