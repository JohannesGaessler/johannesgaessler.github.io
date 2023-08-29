for q in f16 q8_0
do
    sbatch --gpus=p40:1 ./ppl_sbatch.sh $q
done
for q in q4_0 q4_1 q5_0 q5_1 q2_k q3_k_s q3_k_m q3_k_l q4_k_s q4_k_m q5_k_s q5_k_m q6_k
do
    sbatch --gpus=1 --job-name=$q ./ppl_sbatch.sh $q
done
