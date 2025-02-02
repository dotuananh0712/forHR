import pandas as pd
import numpy as np

# Read in csv
business = pd.read_csv(
    'Yelp/yelp-dataset/yelp_academic_dataset_business.csv')
review = pd.read_csv('Yelp/yelp-dataset/yelp_academic_dataset_review.csv')
user = pd.read_csv('Yelp/yelp-dataset/yelp_academic_dataset_user.csv')

# Add col names to datasets
business.columns = ['business_id', 'city', 'business_name',
                    'categories', 'review_count', 'avg_stars']
review.columns = ['user_id', 'business_id', 'stars']
user.columns = ['user_id', 'user_name']

# Join datasets
df1 = pd.merge(left=review, right=user, on='user_id')
df = pd.merge(left=df1, right=business, on='business_id')

# Add is_indian column for any review that has 'Indian' in 'categories'
df['is_indian'] = df['categories'].apply(
    lambda x: True if 'Indian' in x else False)
indian = df[df['is_indian'] == True]

# Search for Indian name
file = open('Yelp/indian_names.txt', 'r')
indian_names = []
for line in file:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    indian_names.append(line_list)

file.close()

mask = np.column_stack([indian['user_name'].isin(indian_names) for col in indian])
result_rows = indian.loc[mask.any(axis=1)]
