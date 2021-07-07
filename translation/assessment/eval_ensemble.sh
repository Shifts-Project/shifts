#!/bin/bash

# Specify path to output of fairseq-generate. Assumed ./ensemble/results_testX.txt
output_dir=ensemble

(
  cd ${output_dir}
  # Get hypotheses and eval
  for i in test1 test2 test3 test4; do
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

    # Get all the uncertainties and process them separately
    grep "^SU-*" results-${i}.txt | grep -v "SUNL" >${i}/tmp
    awk '{print $2}' ${i}/tmp >${i}/entropy_expected.txt
    awk '{print $3}' ${i}/tmp >${i}/expected_entropy.txt
    awk '{print $4}' ${i}/tmp >${i}/mutual_information.txt
    awk '{print $5}' ${i}/tmp >${i}/epkl.txt
    awk '{print $6}' ${i}/tmp >${i}/score.txt
    awk '{print $7}' ${i}/tmp >${i}/aep_tu.txt
    awk '{print $8}' ${i}/tmp >${i}/aep_du.txt
    awk '{print $9}' ${i}/tmp >${i}/npmi.txt
    awk '{print $10}' ${i}/tmp >${i}/score_npmi.txt
    awk '{print $11}' ${i}/tmp >${i}/log_probs.txt
    awk '{print $12}' ${i}/tmp >${i}/ep_entropy_expected.txt
    awk '{print $13}' ${i}/tmp >${i}/ep_mutual_information.txt
    awk '{print $14}' ${i}/tmp >${i}/ep_epkl.txt
    awk '{print $15}' ${i}/tmp >${i}/ep_mkl.txt
    awk '{print $16}' ${i}/tmp >${i}/mkl.txt

    # Heuric Measures of uncertainty
    awk '{print $17}' ${i}/tmp >${i}/var.txt
    awk '{print $18}' ${i}/tmp >${i}/varcombo.txt
    awk '{print $19}' ${i}/tmp >${i}/logvar.txt
    awk '{print $20}' ${i}/tmp >${i}/logcombo.txt

    # Clean up
    rm ${i}/hypos_tmp.txt ${i}/refs_tmp.txt
  done
)

#Evaluate submission
python shifts/translation/assessment/evaluate.py ensemble/test ensemble/test1 --save_path ./results-ensemble.txt --beam_width 5 --nbest 5 --ensemble

# Prepare submission
python shifts/translation/assessment/create_submission.py ensemble/test ensemble/test1 --save_path ./submission-ensemble.json --beam_width 5 --nbest 5 --ensemble --uncertainty_metric SCR-PE
