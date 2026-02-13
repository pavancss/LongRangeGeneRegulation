#! /bin/bash
#SBATCH --job-name=PCNGS12_DMSO     # create a short name for your job
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --ntasks=1                  # Number of tasks (i.e. processes)
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G         
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=pc2976@princeton.edu
#SBATCH --output=PCNGS12_DMSO_control.SLURM_output-%x.%j.txt


module purge

module load samtools 


source activate pairtools 

samtools view -h bam/29NO_DMSO.bam | pairtools parse --nproc-in 50 --nproc-out 50 -c microC/dm6_index/dm6.chrom.sizes -o pairs/29NO_DMSO_parsed.pairsam.gz

echo 'parse done'

pairtools sort --nproc 64 --tmpdir tmp1 -o pairs/29NO_DMSO_sorted.pairsam.gz pairs/29NO_DMSO_parsed.pairsam.gz

echo 'sort done'

pairtools dedup -p 64 --mark-dups --output-dups pairs/29NO_DMSO_duplicates.pairsam.gz  -o pairs/29NO_DMSO_deduped.pairsam.gz pairs/29NO_DMSO_sorted.pairsam.gz

echo 'deduplication done'

pairtools select --nproc-in 64 --nproc-out 64 '(pair_type == "UU") or (pair_type == "UR") or (pair_type == "RU")' -o pairs/29NO_DMSO_filtered.pairsam.gz pairs/29NO_DMSO_deduped.pairsam.gz

echo "select done"

pairtools split --nproc-in 64 --nproc-out 64 --output-pairs pairs/29NO_DMSO_output.pairs.gz pairs/29NO_DMSO_filtered.pairsam.gz

echo "split done"

pairix -f pairs/29NO_DMSO_output.pairs.gz

echo "pairix done"

mamba deactivate


source activate cooler


cooler cload pairix -p 64 microC/dm6_index/dm6.chrom.sizes:100 pairs/29NO_DMSO_output.pairs.gz cools/29NO_DMSO.cool

echo ".cool made"

cooler zoomify -p 64 --balance cools/29NO_DMSO.cool