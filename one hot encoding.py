import numpy as np

# Sample categorical data
X = np.array(["Delhi", "Mumbai", "Delhi", "Chennai"])

# Step 1: Get unique categories
categories = np.unique(X)

# Step 2: Create mapping from category to column index
category_to_index = {}
for idx, category in enumerate(categories):
    category_to_index[category] = idx

# Step 3: Create zero matrix
one_hot = np.zeros((len(X), len(categories)))

# Step 4: Fill matrix using loop
for i, value in enumerate(X):
    col_index = category_to_index[value]
    one_hot[i, col_index] = 1

print("Categories:", categories)
print("One Hot Encoded Matrix:\n", one_hot)
