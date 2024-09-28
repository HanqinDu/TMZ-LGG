#!/usr/bin/python
#-*- coding:utf-8 -*-

from configparser import ConfigParser
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import glob
import json
import re

config = ConfigParser()
config.read(os.path.join("config", "config.ini"))

def mergeClinical():
    # load path
    PATH_CLINICAL = config.get('paths', 'clinical').replace("/", os.sep)
    PATH_SPECIMEN = config.get('paths', 'specimen').replace("/", os.sep)
    PATH_DRUG_CORRECTION = config.get('paths', 'drug_correction').replace("/", os.sep)
    PATH_MERGED_CLINICAL_SPECIMEN = config.get('paths', 'merged_clinical').replace("/", os.sep)

    # Load Clinial and Specimen Data
    df_correction = pd.read_csv(PATH_DRUG_CORRECTION)
    df_correction.columns = ['drug_name', 'correction']

    df_clinical = pd.read_csv(PATH_CLINICAL, sep='\t', header=1)
    df_clinical = df_clinical.drop([0])

    df_specimen = pd.read_csv(PATH_SPECIMEN, sep='\t', header=1)
    df_specimen = df_specimen.drop([0])

    # Correction, Drop Duplicate and Filter
    ## drug name correction
    df_clinical = pd.merge(df_clinical, df_correction, how='left', on='drug_name')
    df_clinical["drug_name"] = df_clinical["correction"]
    df_clinical = df_clinical.drop(["correction"], axis=1)

    ## drop duplicate clinical record
    df_clinical = df_clinical[['bcr_patient_barcode', 'drug_name', 'measure_of_response', 'days_to_drug_therapy_start',
                               'days_to_drug_therapy_end']]
    df_clinical = df_clinical.drop_duplicates()

    ## retain only the patient with valid procurement record
    df_specimen = df_specimen[[s.lstrip("-").isnumeric() for s in df_specimen["days_to_sample_procurement"]]]

    ## when one patient have multiple procurement, retain the earlier one
    df_specimen = df_specimen.astype({"days_to_sample_procurement": "int"})
    df_specimen = df_specimen.sort_values(by=["days_to_sample_procurement"])
    df_specimen = df_specimen[[not dup for dup in df_specimen['bcr_patient_barcode'].duplicated()]]

    ## merge clinical and specimen data
    df_clinical_specimen = df_clinical.merge(df_specimen, how='inner', on=['bcr_patient_barcode'])

    ## Drop Patient with invalid record of drug name, drug response, days to tumor procurement and days to tumor therapy start
    df_clinical_specimen = df_clinical_specimen.dropna(
        subset=['bcr_patient_barcode', 'drug_name', 'measure_of_response', 'days_to_drug_therapy_start',
                'days_to_sample_procurement'])
    index = df_clinical_specimen[
        (df_clinical_specimen['measure_of_response'] == '[Not Applicable]') |
        (df_clinical_specimen['measure_of_response'] == '[Not Available]') |
        (df_clinical_specimen['measure_of_response'] == '[Unknown]') |
        (df_clinical_specimen['measure_of_response'] == '[Discrepancy]') |
        (df_clinical_specimen['drug_name'] == '[Not Available]') |
        (df_clinical_specimen['drug_name'] == 'Not specified') |
        (df_clinical_specimen['days_to_drug_therapy_start'] == '[Discrepancy]') |
        (df_clinical_specimen['days_to_drug_therapy_start'] == '[Not Available]')].index
    df_clinical_specimen = df_clinical_specimen.drop(index)

    ## Filter patient who received drug treatments before tumor resections
    df_clinical_specimen = df_clinical_specimen[
        (df_clinical_specimen['days_to_drug_therapy_start']).astype(int) >=
        (df_clinical_specimen['days_to_sample_procurement']).astype(int)]
    df_clinical_specimen = df_clinical_specimen.sort_values(by=['bcr_patient_barcode'])

    # save merged clinical data
    df_clinical_specimen.to_csv(PATH_MERGED_CLINICAL_SPECIMEN, index=False)



def filterClincial():
    # load path
    PATH_MERGED_CLINICAL = config.get('paths', 'merged_clinical').replace("/", os.sep)
    PATH_FILTERED_MERGED_CLINICAL = config.get('paths', 'merged_filtered_clinical').replace("/", os.sep)

    # load data
    df_clinical_biospecimen = pd.read_csv(PATH_MERGED_CLINICAL)
    df_clinical_biospecimen = df_clinical_biospecimen.groupby(["bcr_patient_barcode", "drug_name"]).filter(
        lambda x: len(x["measure_of_response"].unique()) == 1)

    # When there are multiple record for a patient-drug pair, keep the earlier one
    df_clinical_biospecimen = df_clinical_biospecimen.sort_values(by='days_to_drug_therapy_start')
    df_clinical_biospecimen = df_clinical_biospecimen.groupby(
        ["bcr_patient_barcode", "drug_name"]).first().reset_index()

    # Save Filtered Clinical Data
    df_clinical_biospecimen.to_csv(PATH_FILTERED_MERGED_CLINICAL, index=False)



class Filter(object):

    def __init__(self):
        self.filter = {"op": "and","content": []}

    def add_filter(self, Field, Value, Operator):
        self.filter['content'].append({"op":Operator,"content":{"field":Field,"value":Value}})

    def create_filter(self):
        self.final_filter = json.dumps(self.filter,separators=(',',':'))
        return self.final_filter


def is_tumour_barcode(barcode):
    if (len(barcode) == 0):
        return (False)
    try:
        return ((int)(re.search("^[0-9]*", barcode.split("-")[3])[0]) < 10)
    except:
        print("invalid barcode: " + barcode)
        return (False)


def get_meta(manifest, two_submitter=False):
    uuid_list = pd.Series.tolist(manifest["id"])

    File_Filter = Filter()
    File_Filter.add_filter("files.file_id", uuid_list, "in")
    # File_Filter.add_filter("files.analysis.workflow_type",["STAR - Counts"],"in")
    File_Filter.create_filter()

    data_endpt = "https://api.gdc.cancer.gov/files"
    Fields = ["cases.samples.portions.analytes.aliquots.submitter_id",
              "file_name", "cases.samples.sample_type",
              "file_id",
              "md5sum",
              "experimental_strategy", "analysis.workflow_type",
              "data_type"]
    Fields = ",".join(Fields)

    params = {'filters': File_Filter.create_filter(), "fields": Fields, "format": "json", "size": "10000"}

    response = requests.post(data_endpt, headers={"Content-Type": "application/json"}, json=params)

    data = json.loads(response.text)

    file_list = data['data']['hits']

    Dictionary = []
    TCGA_Barcode_Dict = {}
    for file in file_list:
        UUID = file['file_id']
        Barcode = file['cases'][0]['samples'][0]['portions'][0]['analytes'][0]['aliquots'][0]['submitter_id']

        try:
            Barcode2 = file['cases'][0]['samples'][1]['portions'][0]['analytes'][0]['aliquots'][0]['submitter_id']
        except:
            Barcode2 = ""

        File_Name = file['file_name']

        if (not is_tumour_barcode(Barcode) and is_tumour_barcode(Barcode2)):
            Dictionary.append(
                [UUID, File_Name, Barcode2, Barcode, file['md5sum'], file['cases'][0]['samples'][0]['sample_type'],
                 file['experimental_strategy'], file['analysis']['workflow_type'], file['data_type']])
            TCGA_Barcode_Dict[File_Name] = {Barcode2}
        else:
            Dictionary.append(
                [UUID, File_Name, Barcode, Barcode2, file['md5sum'], file['cases'][0]['samples'][0]['sample_type'],
                 file['experimental_strategy'], file['analysis']['workflow_type'], file['data_type']])
            TCGA_Barcode_Dict[File_Name] = {Barcode}

    Dictionary = pd.DataFrame(Dictionary, columns=["uuid", "filename", "barcode", "barcode2", "md5", "sample_type",
                                                   "experimental_strategy", "workflow_type", "data_type"])

    return (Dictionary, TCGA_Barcode_Dict)


    def get_bcr_patient_barcode(barcode):
        try:
            return (re.search("^[0-9a-zA-Z]+-[0-9a-zA-Z]+-[0-9a-zA-Z]+", barcode)[0])
        except:
            return ("")

    def download_molecular(manifest, data_folder):

        # read manifest

        uuid_list = pd.Series.tolist(manifest["id"])

        # create target dir
        if (not os.path.exists(data_folder)):
            os.mkdir(data_folder)

        # download data by HTTP request - GET
        for i in uuid_list:
            try:
                file_id = i
                data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)

                response = requests.get(data_endpt, headers={"Content-Type": "application/json"})

                # The file name can be found in the header within the Content-Disposition key.
                response_head_cd = response.headers["Content-Disposition"]

                file_name = re.findall("filename=(.+)", response_head_cd)[0]

                with open(os.path.join(data_folder, file_name), "wb") as output_file:
                    output_file.write(response.content)
            except:
                print("fail to download molecular file with uuid: " + i)


    def read_molecular(file_dir, barcode_dict, column_id, column_value, filter_list={}, header=0):
        file_names = os.listdir(file_dir)

        # read and merge data from all files in the folder
        for i, file in enumerate(file_names):
            name = str(list(barcode_dict[file])[0])
            df = pd.read_csv(os.path.join(file_dir, file), sep='\t', header=header)
            for column_filter, value_filter in filter_list:
                df = df[df[column_filter] == value_filter]
            df2 = df[[column_id, column_value]]
            df2.columns = [column_id, name]
            if (i == 0):
                df_output = df2
            else:
                df_output = df_output.merge(df2, how='outer', on=[column_id])

        # reshape
        df_output = df_output.transpose()
        df_output = df_output.rename(columns=df_output.iloc[0]).drop(df_output.index[0])
        df_output = df_output.sort_index(axis=1)
        df_output = df_output.sort_index(axis=0)
        df_output = df_output.reset_index()
        df_output.rename(columns={'index': 'Patient'}, inplace=True)

        return df_output


if __name__=='__main__':
    MANIFEST_MRNA = config.get('paths', 'manifest_mRNA').replace("/", os.sep)
    MANIFEST_MIRNA = config.get('paths', 'manifest_miRNA').replace("/", os.sep)
    MANIFEST_ISOMIR = config.get('paths', 'manifest_isomiR').replace("/", os.sep)
    MANIFEST_METHY = config.get('paths', 'manifest_methy').replace("/", os.sep)
    MANIFEST_CNV = config.get('paths', 'manifest_CNV').replace("/", os.sep)

    META_MRNA = config.get('paths', 'meta_mRNA').replace("/", os.sep)
    META_MIRNA = config.get('paths', 'meta_miRNA').replace("/", os.sep)
    META_ISOMIR = config.get('paths', 'meta_isomiR').replace("/", os.sep)
    META_METHY = config.get('paths', 'meta_methy').replace("/", os.sep)
    META_CNV = config.get('paths', 'meta_CNV').replace("/", os.sep)

    PATH_DRUG_CORRECTION = config.get('paths', 'drug_correction').replace("/", os.sep)
    PATH_MERGED_CLINICAL_SPECIMEN = config.get('paths', 'merged_clinical').replace("/", os.sep)


    if(not os.path.isfile(PATH_DRUG_CORRECTION)):
        mergeClinical()

    if(not os.path.isfile(PATH_MERGED_CLINICAL_SPECIMEN)):
        filterClincial()




    # Load Manifest from GDC
    manifest_mRNA = pd.read_csv(MANIFEST_MRNA, sep='\t')
    manifest_miRNA = pd.read_csv(MANIFEST_MIRNA, sep='\t')
    manifest_isomiR = pd.read_csv(MANIFEST_ISOMIR, sep='\t')
    manifest_methy = pd.read_csv(MANIFEST_METHY, sep='\t')
    manifest_CNV = pd.read_csv(MANIFEST_CNV, sep='\t')

    # Download meta info
    meta_mRNA, barcode_dict_mRNA = get_meta(manifest_mRNA)
    meta_miRNA, barcode_dict_miRNA = get_meta(manifest_miRNA)
    meta_isomiR, barcode_dict_isomiR = get_meta(manifest_isomiR)
    meta_methy, barcode_dict_methy = get_meta(manifest_methy)
    meta_CNV, barcode_dict_CNV = get_meta(manifest_CNV)

    # save meta
    meta_mRNA.to_csv(META_MRNA, index=False)
    meta_miRNA.to_csv(META_MIRNA, index=False)
    meta_isomiR.to_csv(META_ISOMIR, index=False)
    meta_methy.to_csv(META_METHY, index=False)
    meta_CNV.to_csv(META_CNV, index=False)

    # load filtered clincial data
    df_clinical_biospecimen = pd.read_csv(PATH_MERGED_CLINICAL_SPECIMEN)
    df_clinical_biospecimen

    # get bcr_patient_barcode from metadata
    meta_mRNA.insert(loc=0, column='bcr_patient_barcode',
                     value=[get_bcr_patient_barcode(b) for b in meta_mRNA["barcode"]])
    meta_miRNA.insert(loc=0, column='bcr_patient_barcode',
                      value=[get_bcr_patient_barcode(b) for b in meta_miRNA["barcode"]])
    meta_isomiR.insert(loc=0, column='bcr_patient_barcode',
                       value=[get_bcr_patient_barcode(b) for b in meta_isomiR["barcode"]])
    meta_methy.insert(loc=0, column='bcr_patient_barcode',
                      value=[get_bcr_patient_barcode(b) for b in meta_methy["barcode"]])
    meta_CNV.insert(loc=0, column='bcr_patient_barcode',
                    value=[get_bcr_patient_barcode(b) for b in meta_CNV["barcode"]])

    # filter metadata to keep only the patient in clinical data
    target_patients = df_clinical_biospecimen["bcr_patient_barcode"].array

    meta_mRNA = meta_mRNA.loc[[barcode in target_patients for barcode in meta_mRNA["bcr_patient_barcode"]]]
    meta_miRNA = meta_miRNA.loc[[barcode in target_patients for barcode in meta_miRNA["bcr_patient_barcode"]]]
    meta_isomiR = meta_isomiR.loc[[barcode in target_patients for barcode in meta_isomiR["bcr_patient_barcode"]]]
    meta_methy = meta_methy.loc[[barcode in target_patients for barcode in meta_methy["bcr_patient_barcode"]]]
    meta_CNV = meta_CNV.loc[[barcode in target_patients for barcode in meta_CNV["bcr_patient_barcode"]]]

    # filter manifest by filtered metadata
    manifest_mRNA = manifest_mRNA.loc[[i in meta_mRNA["uuid"].array for i in manifest_mRNA["id"]]]
    manifest_miRNA = manifest_miRNA.loc[[i in meta_miRNA["uuid"].array for i in manifest_miRNA["id"]]]
    manifest_isomiR = manifest_isomiR.loc[[i in meta_isomiR["uuid"].array for i in manifest_isomiR["id"]]]
    manifest_methy = manifest_methy.loc[[i in meta_methy["uuid"].array for i in manifest_methy["id"]]]
    manifest_CNV = manifest_CNV.loc[[i in meta_CNV["uuid"].array for i in manifest_CNV["id"]]]


    # download molecular file
    if (not os.path.exists("download")):
        os.mkdir("download")
    download_molecular(manifest_mRNA, FOLDER_DOWNLOAD_MRNA)
    download_molecular(manifest_miRNA, FOLDER_DOWNLOAD_MIRNA)
    download_molecular(manifest_isomiR, FOLDER_DOWNLOAD_ISOMIR)
    download_molecular(manifest_methy, FOLDER_DOWNLOAD_METHY)
    download_molecular(manifest_CNV, FOLDER_DOWNLOAD_CNV)


    # read molecular data
    df_mRNA_fpkm = read_molecular(
        file_dir=FOLDER_DOWNLOAD_MRNA,
        barcode_dict=barcode_dict_mRNA,
        column_id="gene_id",
        column_value="fpkm_unstranded",
        filter_list={("gene_type", "protein_coding")},
        header=1)

    df_mRNA_fpkm_uq = read_molecular(
        file_dir=FOLDER_DOWNLOAD_MRNA,
        barcode_dict=barcode_dict_mRNA,
        column_id="gene_id",
        column_value="fpkm_uq_unstranded",
        filter_list={("gene_type", "protein_coding")},
        header=1)

    df_miRNA = read_molecular(
        file_dir=FOLDER_DOWNLOAD_MIRNA,
        barcode_dict=barcode_dict_miRNA,
        column_id="miRNA_ID",
        column_value="reads_per_million_miRNA_mapped")

    df_isomiR = read_molecular(
        file_dir=FOLDER_DOWNLOAD_ISOMIR,
        barcode_dict=barcode_dict_isomiR,
        column_id="isoform_coords",
        column_value="reads_per_million_miRNA_mapped")

    df_methy = read_molecular(
        file_dir=FOLDER_DOWNLOAD_METHY,
        barcode_dict=barcode_dict_methy,
        column_id=0,
        column_value=1,
        header=-1)

    df_CNV = read_molecular(
        file_dir=FOLDER_DOWNLOAD_CNV,
        barcode_dict=barcode_dict_CNV,
        column_id="gene_id",
        column_value="copy_number")

    # save merged data
    if (not os.path.exists("molecular")):
        os.mkdir("molecular")

    df_mRNA_fpkm.to_feather(PROFILE_MRNA_FPKM)
    df_mRNA_fpkm_uq.to_feather(PROFILE_MRNA_FPKM_UQ)
    df_miRNA.to_feather(PROFILE_MIRNA)
    df_isomiR.to_feather(PROFILE_ISOMIR)
    df_methy.to_feather(PROFILE_METHY)
    df_CNV.to_feather(PROFILE_CNV)










