conda activate TMZ-LGG

python src/evaluate_model.py \
-i data/dataset/omics_1110/dm_miRNA.feather \
-o data/prediction/XGB_default_miRNA.pickle \
-c XGB-C-D \
-s 128 46 59124 1889 6 \
-f 5 \
-j 5 \
-a

