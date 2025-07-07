<head>
<style>
    h1, h2, h3, h4, h5, h6, ul, ol, p {
        font-family: 'Liberation Serif', serif;
    }
    .small {font-size: 5pt; color:#1b76d0;}
    .modified {color:#1b76d0;}
    .department{line-height:1.0; font-size:17pt; text-align:center;}
    .maintitle{line-height:1.3; font-size:23pt; font-weight: bold; text-align:center;}
    .announcement{line-height:2.8; font-size:17pt; font-weight: bold; text-align:center;}
    .author{line-height:1; font-size:17pt; text-align:center;}
    .annotation{line-height:1.1; font-size:13pt; font-weight: italic; text-align:center;}
    .caption{line-height: 1.2; font-size: 11pt;}
    .chapter{font-size:21pt; font-weight: bold; margin-top: 3.5em}
    .pagecenter{font-size:30pt; font-weight: bold; margin-top: 10.0em; text-align:center;}
    .transparent{color: rgba(0, 0, 0, 0); font-size:0pt}
    p{font-size: 12pt; line-height: 1.5; text-align: justify; margin-bottom: 1.3em;}
    h1{font-size:24pt; line-height:1.2; border-bottom: 0px; margin-bottom: 2.4em; margin-top: 1.4em}
    h2{font-size:18pt; line-height:1.5; border-bottom: 0px; margin-top: 1.5em}
    h3{font-size:14pt; line-height:1.2; margin-top: 1.2em}
    ul, ol{font-size: 12pt;}
    table{font-size: 11px;}
    .footnote-line{font-size: 12px;text-align: justify;}
    sup.md-footnote {
        background-color: initial;
        color: inherit;
        margin-left: -3px; 
        margin-right: -3px; 
        font-size: 8pt;
    }
</style>
</head>

This repository contains code for reproducing the key results of the study:

## ** Predicting temozolomide response in low-grade glioma patients with large-scale machine learning 
Hanqin Du¹, Chayanit Piyawajanusorn¹, Ghita Ghislat²\*, Pedro J. Ballester¹\*  
¹ Department of Bioengineering, Imperial College London, UK  
² Department of Life Sciences, Imperial College London, UK  
\*Corresponding authors: ghita.ghislat@imperial.ac.uk, p.ballester@imperial.ac.uk


### Abstract
Background: Temozolomide is the primary chemotherapeutic agent and first-line treatment for low-grade glioma (LGG). Although LGGs are generally less aggressive than high-grade gliomas (HGGs), they can eventually progress into HGGs, making it crucial to maximise the efficacy of initial LGG treatment. Methods: In this study, we analysed data from 109 LGG patients in The Cancer Genome Atlas (TCGA) to assess the predictive performance of 12 machine learning (ML) classification algorithms in forecasting temozolomide response, utilising six types of omics data. Cross-validation and bootstrapping bias correction were employed to compare the performance of these models with that of a conventional biomarker-based model using MGMT promoter methylation status. Results: Among the models, the miRNA-based approach using the XGBoost algorithm showed the most promising predictive performance, with a Matthews Correlation Coefficient (MCC) of 0.447, outperforming the auto-ML method JADBio (MCC = 0.250) and the MGMT biomarker-based model. Converting the task into a regression framework by encoding cancer response as a continuous variable generally weakened predictive performance, with the best model based on methylation profile (MCC = 0.344). Additionally, ML models focusing on biomarker-relevant features demonstrated improved prediction accuracy, though not to the extent of omics-based models, with the support vector machine (SVM) model achieving an MCC of 0.331. Incorporating clinical variables, such as patient age and Karnofsky score, further enhanced predictive power, with the LR-OMC model achieving the highest MCC (0.483). Feature importance analysis identified six significant miRNA factors, including three tumour-related miRNAs (miR-335, let-7f, and miR-7-2) and three potential biomarkers (miR-204, miR-6513, and miR-376). Discussion: Overall, this study systematically demonstrates the potential of large-scale analyses combining machine learning and omics data to predict temozolomide response, offering superior predictive accuracy compared to standard MGMT biomarkers.

### The provided code includes
- `1_download_molecular_profile.sh`: Bash script for downloading raw data from TCGA and constructing it into an ML-ready dataset 
- `2_train_miRNA.sh`: Bash script for training and evaluating the best-performed model in the study. (This will save the produce prediction files)
- `3_analysis_performance.ipynb`: Jupyter notebook file that evaluates the prediction and calculates metrics such as MCC, ROC-AUC and PR-AUC. 



---
