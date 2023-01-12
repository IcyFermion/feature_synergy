#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from argparse import ArgumentParser
import sys
import glob
import csv


args = ArgumentParser('./normalize_rna_counts.py', description="""This program has been designed to
take a table of counts per gene (or feature) from RNA-seq and normalize the counts. If a genome
features file (-g option) containing information about transcript lengths is provided, the default normalization
is read counts per kilobase per million/fragments per kilobase per million (RPKM/FPKM), but can be
changed to transcripts per million (TPM) by using the -t or --tpm option. If no genome file is provided,
counts will be normalized to read counts per million (RCPM). Use the -r option to provide a list of features that
you do not want to be counted for normalization (excluding lines starting with either 'N_' or '__' added by HTSEq-count
or STAR).
Example usage: python normalize_rna_counts.py -c all_samples_counts.csv -g GENCODE_Longest_CDS.csv -t -o All_samples
-r remove.txt""")


args.add_argument(
	'-c',
	'--counts_file',
	help="""This is a file that contains counts per gene (or feature) for all samples. This must be
	in csv format with the first row being sample names and the first column being feature names. """,
	default=None,
)

args.add_argument(
	'-g',
	'--genome_features',
	help="""This is a file that contains information about the features contained in your dataset.
	The column containing the feature name that matches the first column in the --counts_file must
	be labeled as 'gene_id' and the column containing the length of the feature must be labeled as
	'CDS Length'. Instructions on how to generate the genome_features file from a GTF file can be
	found here: https://github.com/davidwsant/parse_gtf """,
	default = None
)

args.add_argument(
	'-t',
	'--tpm',
	help="""Specify this option if you would like to scale using TPM rather than RPKM/FPKM. """,
	action= 'store_true'
)

args.add_argument(
	'-o',
	'--output_prefix',
	help="""You can use this option to specify a prefix that you would like added to the normalized
	output file. By default, the prefix will 'Normalized'. """,
	default = None
)

args.add_argument(
	'-r',
	'--remove',
	help="""Use this option to supply a file with specific feature names that you want to be excluded
	from the total cell reads for normalization. Do not include in this file features added by
	HTSeq-count that begin with '__' or features added by STAR that begin with 'N_' as they will already
	be removed. Features that you may want to include are those that were identified as outliers during
	differential analysis by a method such as Cook's cutoff.""",
	default = None
)

args = args.parse_args()


def print_error_message():
	print()
	print("\tWelcome to normalize_rna_counts.py.")
	print("\tThis program has been designed to take a table of counts per gene (or feature) from RNA-seq ")
	print("\tand normalize the counts. If a genome features file (-g option) containing information about ")
	print("\ttranscript lengths is provided, the default normalization is read counts per kilobase per ")
	print("\tmillion/fragments per kilobase per million (RPKM/FPKM), but can be changed to transacripts ")
	print("\tper million (TPM) by using the -t or --tpm option. If no genome file is provided, counts ")
	print("\twill be normalized to read counts per million (RCPM). ")
	print()
	print("\tExample usage: python normalize_rna_counts.py -c all_samples_counts.csv")
	print("\t-g GENCODE_Longest_CDS.csv -t -o All_samples")
	print()


counts_file = args.counts_file
genome_features = args.genome_features
tpm = args.tpm
output_prefix = args.output_prefix
remove = args.remove


if not counts_file:
	files = glob.glob('*.csv')
	print_error_message()
	print("\tYou have not specified a counts file that you wish to normalize.")
	print("\tPlease specify this file using the -c option.")
	print()
	print("\tPossible files from your current working directory are: ")
	for file in files:
		print('\t'+file)
	print()
	sys.exit(1) # Exit with a status of 1.

if not counts_file.endswith('csv'):
	files = glob.glob('*.csv')
	print_error_message()
	print("\tThe counts file that you have specified does not appear to be a csv file.")
	print("\tPlease specify this file using the -c option.")
	print()
	print("\tPossible files from your current working directory are: ")
	for file in files:
		print('\t'+file)
	print()
	sys.exit(1) # Exit with a status of 1.

counts_df = pd.read_csv(counts_file, index_col = 0, header = 0)

present_features = counts_df.index
if remove:
	remove_features = [line.rstrip('\n') for line in open(remove)]
	not_present_features = []
	for feature in remove_features:
		if feature not in present_features:
			not_present_features.append(feature)

	if len(not_present_features) > 0:
		print()
		print("\tOne or more of the features in the file provided by the -r option were not present in the dataset.")
		print("\tThe following features cannot be removed for normalization:")
		for feature in not_present_features:
			print("\t"+feature)
			remove_features.remove(feature)
		print()

non_scalable_features = []
for feature in present_features:
	if feature.startswith('N_') or feature.startswith('__'):
		non_scalable_features.append(feature)

if len(non_scalable_features) > 0:
	counts_df = counts_df.drop(non_scalable_features, axis = 0)
if remove:
	if len(remove_features) > 0:
		cleaned_df = counts_df.drop(remove_features, axis = 0)
else:
	cleaned_df = counts_df
sum_reads_dict = dict(cleaned_df.sum())

if not genome_features:
	print()
	print('\tNormalizing to Read Counts Per Million mapped reads (RCPM)')
	print()
	# I am not using a function in Pandas because it has been rounding numbers down, but I want to keep the float
	rcpm_dict = {}
	for gene, row in counts_df.iterrows():
		for sample, count in zip(row.index, row):
			if sample not in rcpm_dict:
				rcpm_dict[sample] = {}
			sample_reads = sum_reads_dict[sample]
			rcpm = 1000000*(count/sample_reads)
			rcpm_dict[sample][gene] = rcpm

	rcpm_df = pd.DataFrame(rcpm_dict)
	output_name = 'RCPM_table.csv'
	if output_prefix:
		output_name = output_prefix+'_RCPM_table.csv'
	rcpm_df.to_csv(output_name)
	sys.exit(0)

# Read in the genome features file and get the CDS length per each transcript
# I think this will be faster to use csv library rather than Pandas
features_dict = {}
with open(genome_features, 'r') as file:
	reader = csv.DictReader(file)
	for row in reader:
		features_dict[row['gene_id']] = int(row['CDS Length'])


if tpm:
	print()
	print('\tNormalizing to Transcripts Per Million mapped reads (TPM)')
	print()
	tpm_scaling_dict = {}
	for gene, row in counts_df.iterrows():
		for sample, count in zip(row.index, row):
			if sample not in tpm_scaling_dict:
				tpm_scaling_dict[sample] = {}
			feature_length = features_dict[gene]
			tpm_scaling_factor = 1000*count/feature_length
			tpm_scaling_dict[sample][gene] = tpm_scaling_factor
	tpm_sums = {}
	for sample in tpm_scaling_dict:
		tpm_factor_sum = 0
		for gene in tpm_scaling_dict[sample]:
			tpm_factor_sum += tpm_scaling_dict[sample][gene]
		tpm_sums[sample] = tpm_factor_sum
	tpm_results_dict = {}
	for sample in tpm_scaling_dict:
		tpm_results_dict[sample] = {}
		tpm_sum = tpm_sums[sample]
		for gene in tpm_scaling_dict[sample]:
			tpm = 1000000*tpm_scaling_dict[sample][gene]/tpm_sum
			tpm_results_dict[sample][gene] = tpm

	tpm_df = pd.DataFrame(tpm_results_dict)
	output_name = 'TPM_table.csv'
	if output_prefix:
		output_name = output_prefix+'_TPM_table.csv'
	tpm_df.to_csv(output_name)
	sys.exit(0)

# Everything else will be FPKM
print()
print('\tNormalizing to Reads/Fragments Per Kilobase per Million mapped reads (RPKM/FPKM)')
print()
fpkm_dict = {}
for gene, row in counts_df.iterrows():
	for sample, count in zip(row.index, row):
		if sample not in fpkm_dict:
			fpkm_dict[sample] = {}
		sample_reads = sum_reads_dict[sample]
		feature_length = features_dict[gene]
		fpkm = 1000000000*(count/sample_reads)/feature_length
		fpkm_dict[sample][gene] = fpkm

fpkm_df = pd.DataFrame(fpkm_dict)
output_name = 'FPKM_table.csv'
if output_prefix:
	output_name = output_prefix+'_FPKM_table.csv'
fpkm_df.to_csv(output_name)