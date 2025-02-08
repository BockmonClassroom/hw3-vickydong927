'''
Name: Weiqi Dong
Date: 2/5/2025
'''

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


# Load the datasets
t1 = pd.read_csv('Data/t1_user_active_min.csv')
t2 = pd.read_csv('Data/t2_user_variant.csv')
t3 = pd.read_csv('Data/t3_user_active_min_pre.csv')
t4 = pd.read_csv('Data/t4_user_attributes.csv')


# Part 2: Organizing the Data
# Merge t1 and t2 on 'uid' to combine user activity with variant information
merged_data = pd.merge(t1, t2, on='uid', how='left')
# Group by 'uid' and aggregate the data by summing active minutes
aggregated_data = merged_data.groupby(['uid', 'variant_number'], as_index=False)['active_mins'].sum()
# Rename the columns for clarity
aggregated_data.rename(columns={'active_mins': 'total_active_mins'}, inplace=True)
# Check the aggregated data
print('**************************Part2**************************')
print('Organized data in Data/part2_organize_data.csv file')
print(aggregated_data.head(),'\n')
# Save the aggregated data to a new CSV file
aggregated_data.to_csv('Data/part2_organize_data.csv', index=False)


# Part 3: Statistical Analysis
# Split data into groups
group1 = aggregated_data[aggregated_data['variant_number'] == 0]['total_active_mins']
group2 = aggregated_data[aggregated_data['variant_number'] == 1]['total_active_mins']
# Compute mean and median
# Compute mean, median, and variance
mean1, median1= group1.mean(), group1.median()
mean2, median2= group2.mean(), group2.median()
var1 = group1.var()
var2 = group2.var()
print('**************************Part3**************************')
print(f"Group 1 - Mean: {mean1}, Median: {median1}, Variance: {var1}")
print(f"Group 2 - Mean: {mean2}, Median: {median2}, Variance: {var2}")
# Perform statistical significance test (t-test)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
print("due to different variance between group 1 and group 2, use Welch's t-test for independent samples t-test.")
print(f"T-statistic: {t_stat}, P-value: {p_value}")
# Interpretation
if p_value < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2. \n")
else:
    print("There is no statistically significant difference between Group 1 and Group 2. \n")


# Part 4: Digging a Little Deeper
# Check normality assumption using Kolmogorov-Smirnov test
ks_stat1, ks_p1 = stats.kstest(group1, 'norm', args=(group1.mean(), group1.std()))
ks_stat2, ks_p2 = stats.kstest(group2, 'norm', args=(group2.mean(), group2.std()))
print('**************************Part4**************************')
print(f"Kolmogorov-Smirnov test results for Group 1: KS Statistic = {ks_stat1}, P-value = {ks_p1}")
print(f"Kolmogorov-Smirnov test results for Group 2: KS Statistic = {ks_stat2}, P-value = {ks_p2}")
# Q-Q plot to visualize it
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
stats.probplot(group1, dist="norm", plot=plt)
plt.title("Q-Q Plot for Group 1")
plt.subplot(1, 2, 2)
stats.probplot(group2, dist="norm", plot=plt)
plt.title("Q-Q Plot for Group 2")
plt.show()
# Box plot to visualize distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=[group1, group2], palette=["blue", "orange"])
plt.xticks([0, 1], ["Group 1", "Group 2"])
plt.title("Box Plot of Active Minutes by Group")
plt.ylabel("Total Active Minutes")
plt.show()
# Identify and remove outliers (assuming max active minutes is 1440 per day)
outlier_threshold = 1440
filtered_data = aggregated_data[aggregated_data['total_active_mins'] <= outlier_threshold]
# Redo statistical analysis after removing outliers
group1_filtered = filtered_data[filtered_data['variant_number'] == 0]['total_active_mins']
group2_filtered = filtered_data[filtered_data['variant_number'] == 1]['total_active_mins']
# Compute new mean and median
mean1_filtered, median1_filtered = group1_filtered.mean(), group1_filtered.median()
mean2_filtered, median2_filtered = group2_filtered.mean(), group2_filtered.median()
print(' Redo statistical analysis after removing outliers: outlier_threshold > 1440 ')
print(f"Filtered Group 1 - Mean: {mean1_filtered}, Median: {median1_filtered}")
print(f"Filtered Group 2 - Mean: {mean2_filtered}, Median: {median2_filtered}")
# Re-perform statistical significance test (t-test)
t_stat_filtered, p_value_filtered = stats.ttest_ind(group1_filtered, group2_filtered, equal_var=False)
print(f"Filtered T-statistic: {t_stat_filtered}, P-value: {p_value_filtered}")
if p_value_filtered < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2 after removing outliers.\n")
else:
    print("There is no statistically significant difference between Group 1 and Group 2 after removing outliers.\n")


# Part 5: Digging Even Deeper - Accounting for t3 Data
# Merge t3 data with the previous aggregated data
aggregated_data = aggregated_data.rename(columns={'total_active_mins': 'active_mins_post'})
t3_aggregated = t3.groupby('uid')['active_mins'].sum().reset_index()
t3_aggregated = t3_aggregated.rename(columns={'active_mins': 'active_mins_pre'})
aggregated_data_with_t3 = pd.merge(aggregated_data, t3_aggregated, on='uid', how='left')
# Save the updated data to a new CSV file
aggregated_data_with_t3.to_csv('Data/part5_organize_data_with_t3.csv', index=False)
# Group by 'variant_number' and compute statistics on the post-active minutes
group1_post = aggregated_data_with_t3[aggregated_data_with_t3['variant_number'] == 0]['active_mins_post']
group2_post = aggregated_data_with_t3[aggregated_data_with_t3['variant_number'] == 1]['active_mins_post']
# Compute mean, median, and variance for the post-active minutes
mean1_post, median1_post = group1_post.mean(), group1_post.median()
mean2_post, median2_post = group2_post.mean(), group2_post.median()
var1_post = group1_post.var()
var2_post = group2_post.var()
print('**************************Part5**************************')
print(f"Group 1 (post) - Mean: {mean1_post}, Median: {median1_post}, Variance: {var1_post}")
print(f"Group 2 (post) - Mean: {mean2_post}, Median: {median2_post}, Variance: {var2_post}")
# Perform Welch's t-test (independent t-test with unequal variance)
t_stat_post, p_value_post = stats.ttest_ind(group1_post, group2_post, equal_var=False)
print(f"Post-data T-statistic: {t_stat_post}, P-value: {p_value_post}")
# Interpretation of results
if p_value_post < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2 (post-active minutes).")
else:
    print("There is no statistically significant difference between Group 1 and Group 2 (post-active minutes).")
# Now repeat the same for pre-active minutes
group1_pre = aggregated_data_with_t3[aggregated_data_with_t3['variant_number'] == 0]['active_mins_pre']
group2_pre = aggregated_data_with_t3[aggregated_data_with_t3['variant_number'] == 1]['active_mins_pre']
# Compute mean, median, and variance for the pre-active minutes
mean1_pre, median1_pre = group1_pre.mean(), group1_pre.median()
mean2_pre, median2_pre = group2_pre.mean(), group2_pre.median()
var1_pre = group1_pre.var()
var2_pre = group2_pre.var()
print(f"Group 1 (pre) - Mean: {mean1_pre}, Median: {median1_pre}, Variance: {var1_pre}")
print(f"Group 2 (pre) - Mean: {mean2_pre}, Median: {median2_pre}, Variance: {var2_pre}")
# Perform Welch's t-test for pre-data
t_stat_pre, p_value_pre = stats.ttest_ind(group1_pre, group2_pre, equal_var=False)
print(f"Pre-data T-statistic: {t_stat_pre}, P-value: {p_value_pre}")
# Interpretation of results
if p_value_pre < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2 (pre-active minutes). \n")
else:
    print("There is no statistically significant difference between Group 1 and Group 2 (pre-active minutes).\n")

# Identifying and Removing Outliers (assuming max active minutes per day is 1440)
print(' Redo statistical analysis after removing outliers: outlier_threshold > 1440 ')
outlier_threshold = 1440

# Filter out outliers for 'active_mins_post' and 'active_mins_pre'
filtered_data_post = aggregated_data_with_t3[aggregated_data_with_t3['active_mins_post'] <= outlier_threshold]
filtered_data_pre = aggregated_data_with_t3[aggregated_data_with_t3['active_mins_pre'] <= outlier_threshold]

# Merge the filtered data to ensure consistency
filtered_aggregated_data = pd.merge(filtered_data_post, filtered_data_pre, on='uid', how='left')
# Rename columns to make them consistent
filtered_aggregated_data.rename(columns={
    'variant_number_x': 'variant_number',
    'active_mins_post_x': 'active_mins_post',
    'active_mins_pre_x': 'active_mins_pre',
    'variant_number_y': 'variant_number_pre',  # This will be kept separately for pre-active minutes
    'active_mins_post_y': 'active_mins_post_pre',
    'active_mins_pre_y': 'active_mins_pre_pre'
}, inplace=True)

# Now, re-perform the statistical analysis after removing outliers

# For post-active minutes (filtered data)
group1_post_filtered = filtered_aggregated_data[filtered_aggregated_data['variant_number'] == 0]['active_mins_post']
group2_post_filtered = filtered_aggregated_data[filtered_aggregated_data['variant_number'] == 1]['active_mins_post']

# Compute mean, median, and variance for filtered post-active minutes
mean1_post_filtered, median1_post_filtered = group1_post_filtered.mean(), group1_post_filtered.median()
mean2_post_filtered, median2_post_filtered = group2_post_filtered.mean(), group2_post_filtered.median()

var1_post_filtered = group1_post_filtered.var()
var2_post_filtered = group2_post_filtered.var()
print(f"Filtered Group 1 (post) - Mean: {mean1_post_filtered}, Median: {median1_post_filtered}, Variance: {var1_post_filtered}")
print(f"Filtered Group 2 (post) - Mean: {mean2_post_filtered}, Median: {median2_post_filtered}, Variance: {var2_post_filtered}")

# Perform Welch's t-test (independent t-test with unequal variance) for filtered post-active minutes
t_stat_post_filtered, p_value_post_filtered = stats.ttest_ind(group1_post_filtered, group2_post_filtered, equal_var=False)
print(f"Filtered Post-data T-statistic: {t_stat_post_filtered}, P-value: {p_value_post_filtered}")

# Interpretation of filtered results
if p_value_post_filtered < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2 (filtered post-active minutes).")
else:
    print("There is no statistically significant difference between Group 1 and Group 2 (filtered post-active minutes).")

# For pre-active minutes (filtered data)
group1_pre_filtered = filtered_aggregated_data[filtered_aggregated_data['variant_number'] == 0]['active_mins_pre']
group2_pre_filtered = filtered_aggregated_data[filtered_aggregated_data['variant_number'] == 1]['active_mins_pre']

# Compute mean, median, and variance for filtered pre-active minutes
mean1_pre_filtered, median1_pre_filtered = group1_pre_filtered.mean(), group1_pre_filtered.median()
mean2_pre_filtered, median2_pre_filtered = group2_pre_filtered.mean(), group2_pre_filtered.median()

var1_pre_filtered = group1_pre_filtered.var()
var2_pre_filtered = group2_pre_filtered.var()
print(f"Filtered Group 1 (pre) - Mean: {mean1_pre_filtered}, Median: {median1_pre_filtered}, Variance: {var1_pre_filtered}")
print(f"Filtered Group 2 (pre) - Mean: {mean2_pre_filtered}, Median: {median2_pre_filtered}, Variance: {var2_pre_filtered}")

# Perform Welch's t-test (independent t-test with unequal variance) for filtered pre-active minutes
t_stat_pre_filtered, p_value_pre_filtered = stats.ttest_ind(group1_pre_filtered, group2_pre_filtered, equal_var=False)
print(f"Filtered Pre-data T-statistic: {t_stat_pre_filtered}, P-value: {p_value_pre_filtered}")

# Interpretation of filtered results
if p_value_pre_filtered < 0.05:
    print("There is a statistically significant difference between Group 1 and Group 2 (filtered pre-active minutes).")
else:
    print("There is no statistically significant difference between Group 1 and Group 2 (filtered pre-active minutes).\n")


# Part 6: merge t4
# Merge the datasets on 'uid'
p5_data = pd.read_csv('Data/part5_organize_data_with_t3.csv')
t4_merged_data = pd.merge(p5_data, t4, on='uid', how='inner')
# Save the merged data to a new CSV file
t4_merged_data.to_csv('Data/part6_organize_data_with_t4.csv', index=False)
# Group by gender and calculate mean and median of active minutes (both post and pre)
gender_grouped_data = t4_merged_data.groupby('gender')[['active_mins_post', 'active_mins_pre']].agg(['mean', 'median'])
print('**************************Part6**************************')
print('merge gender feature')
print(gender_grouped_data)
# Separate the data by gender
male_data = t4_merged_data[t4_merged_data['gender'] == 'male']
female_data = t4_merged_data[t4_merged_data['gender'] == 'female']
# Perform t-test for 'active_mins_post'
t_stat_post, p_value_post = ttest_ind(male_data['active_mins_post'], female_data['active_mins_post'], equal_var=False)
# Perform t-test for 'active_mins_pre'
cleaned_data = t4_merged_data.dropna(subset=['active_mins_pre'])
male_data = cleaned_data[cleaned_data['gender'] == 'male']['active_mins_pre']
female_data = cleaned_data[cleaned_data['gender'] == 'female']['active_mins_pre']
t_stat, p_value = ttest_ind(male_data, female_data, equal_var=False)
# Print the t-test results
print("T-test for active_mins_post: T-statistic =", t_stat_post, ", P-value =", p_value_post)
if p_value_post < 0.05:
    print("There is a statistically significant difference in active_mins_post between the groups.")
else:
    print("There is no statistically significant difference in active_mins_post between the groups.")
print("T-test for active_mins_pre: T-statistic =", t_stat, ", P-value =", p_value)
if p_value < 0.05:
    print("There is a statistically significant difference in active_mins_pre between the groups.")
else:
    print("There is no statistically significant difference in active_mins_pre between the groups.")
# Set up the plot style
sns.set(style="whitegrid")
# Create a box plot for active minutes post (grouped by gender)
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='active_mins_post', data=t4_merged_data)
plt.title('Active Minutes Post by Gender')
plt.show()
# Create a box plot for active minutes pre (grouped by gender)
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='active_mins_pre', data=t4_merged_data)
plt.title('Active Minutes Pre by Gender')
plt.show()