python src/evaluate_model.py \
-i data/dataset/omics_1110/dm_miRNA.feather \
-o data/prediction/XGB_miRNA.pickle \
-c XGB \
-j 5 \
-f 5 \
-p 2 \
-r 50 \
-s 128 46 59124 1889 6
