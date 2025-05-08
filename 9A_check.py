import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
mutation_df = pd.read_csv("C:/Users/Zubair/Desktop/Sup/mutations_with_signatures_and_DP.tsv", sep="\t")
clinical_df = pd.read_csv("C:/Users/Zubair/Desktop/Sup/luad_tcga_gdc_clinical_data.tsv", sep="\t")

# Define LUAD-specific driver genes
driver_genes = {
    'TP53', 'EGFR', 'KRAS', 'STK11', 'KEAP1', 'NF1', 'RB1', 'PIK3CA',
    'MET', 'ERBB2', 'ALK', 'ROS1', 'RET', 'BRAF',
    'RBM10', 'U2AF1', 'ARID1A', 'SMARCA4'
}

# Clinical statistics
clinical_stats = {
    "Never smoker (%)": (clinical_df['Person Cigarette Smoking History Pack Year Value'] == 0).mean() * 100,
    "Former smoker (%)": ((clinical_df['Person Cigarette Smoking History Pack Year Value'] > 0) & 
                          (clinical_df["Patient's Vital Status"] != 'Current smoker')).mean() * 100,
    "Current smoker (%)": (clinical_df["Patient's Vital Status"] == 'Current smoker').mean() * 100,
    "Stage I (%)": clinical_df['AJCC Pathologic Stage'].str.contains('I', na=False).mean() * 100,
    "Stage II (%)": clinical_df['AJCC Pathologic Stage'].str.contains('II', na=False).mean() * 100,
    "Stage III (%)": clinical_df['AJCC Pathologic Stage'].str.contains('III', na=False).mean() * 100,
    "Stage IV (%)": (clinical_df['AJCC Pathologic Stage'] == 'Stage IV').mean() * 100,
    "Male (%)": (clinical_df['Sex'] == 'Male').mean() * 100,
    "Female (%)": (clinical_df['Sex'] == 'Female').mean() * 100,
    "Median age at diagnosis": clinical_df['Diagnosis Age'].median(),
    "5-year survival rate (%)": (clinical_df[clinical_df['Overall Survival (Months)'] >= 60]['Overall Survival Status'] == '0:LIVING').mean() * 100
}

# Genomic statistics
genomic_stats = {
    "Median mutations/sample": mutation_df['Tumor_Sample_Barcode'].value_counts().median(),
    "Mean mutations/sample": mutation_df['Tumor_Sample_Barcode'].value_counts().mean(),
    "Driver mutations (%)": (mutation_df['Hugo_Symbol'].isin(driver_genes)).mean() * 100,
    "Fraction genome altered (median)": clinical_df['Fraction Genome Altered'].median() * 100
}

# Technical statistics
technical_stats = {
    "Mean coverage depth": mutation_df['DP'].mean(),
    "Median coverage depth": mutation_df['DP'].median(),
    "% PASS variants": (mutation_df['callers'] == 'PASS').mean() * 100 if 'callers' in mutation_df.columns else "N/A"
}

# Combine all statistics into one table
supp_table1_data = {**clinical_stats, **genomic_stats, **technical_stats}
supp_table1 = pd.DataFrame(supp_table1_data.items(), columns=['Characteristic', 'Value'])
supp_table1.to_csv("C:/Users/Zubair/Desktop/Sup/Supplementary_Table1.csv", index=False)

# Plot Figure 1
plt.figure(figsize=(20, 14))

# 1. Mutation burden per sample
plt.subplot(3, 3, 1)
sns.histplot(mutation_df['Tumor_Sample_Barcode'].value_counts(), bins=50, kde=True)
plt.title("Mutation Burden per Sample\n(Median = {:.0f})".format(genomic_stats["Median mutations/sample"]))
plt.xlabel("Mutations per sample")

# 2. Smoking status pie
plt.subplot(3, 3, 2)
smoking_data = {
    'Never': clinical_stats["Never smoker (%)"],
    'Former': clinical_stats["Former smoker (%)"],
    }
plt.pie(smoking_data.values(), labels=smoking_data.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Smoking Status Distribution")

# 3. Stage distribution
plt.subplot(3, 3, 3)
stage_data = {
    'I': clinical_stats["Stage I (%)"],
    'II': clinical_stats["Stage II (%)"],
    'III': clinical_stats["Stage III (%)"],
    'IV': clinical_stats["Stage IV (%)"]
}
sns.barplot(x=list(stage_data.keys()), y=list(stage_data.values()))
plt.title("Tumor Stage Distribution")
plt.ylabel("Percentage")

# 4. Driver gene landscape
plt.subplot(3, 3, 4)
driver_counts = mutation_df[mutation_df['Hugo_Symbol'].isin(driver_genes)]['Hugo_Symbol'].value_counts().nlargest(10)
sns.barplot(y=driver_counts.index, x=driver_counts.values)
plt.title("Top 10 Driver Genes")
plt.xlabel("Mutation Count")

# 5. Gender distribution pie
plt.subplot(3, 3, 5)
gender_data = {
    'Male': clinical_stats["Male (%)"],
    'Female': clinical_stats["Female (%)"]
}
plt.pie(gender_data.values(), labels=gender_data.keys(), autopct='%1.1f%%', startangle=90)
plt.title("Gender Distribution")

# 6. Fraction genome altered histogram
plt.subplot(3, 3, 6)
sns.histplot(clinical_df['Fraction Genome Altered'].dropna() * 100, bins=30, kde=True)
plt.title("Fraction Genome Altered\n(Median = {:.1f}%)".format(genomic_stats["Fraction genome altered (median)"]))
plt.xlabel("Percentage of Genome Altered")

# 7. SBS Signature distribution
plt.subplot(3, 3, 7)
dominant_sigs = mutation_df['Dominant_Signature'].value_counts().nlargest(10)
sns.barplot(x=dominant_sigs.index, y=dominant_sigs.values)
plt.xticks(rotation=45)
plt.title("Top 10 Dominant SBS Signatures")
plt.ylabel("Mutation Count")

# Final layout & save
plt.tight_layout()
plt.savefig("C:/Users/Zubair/Desktop/Sup/Figure1_updated.png", dpi=1200, bbox_inches='tight')
plt.show()

print("✅ Supplementary Table 1 and updated Figure 1 created successfully!")
print(f"Cohort size: n={len(clinical_df)}")
