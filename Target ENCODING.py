def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    unique_categories = set(categories)
    
    category_mean = {}
    
    for cat in unique_categories:
        total = 0
        count = 0
        
        for i in range(len(categories)):
            if categories[i] == cat:
                total += targets[i]
                count += 1
        
        category_mean[cat] = total / count
        #in target encoder remember it with target we find it by using the target value
    
    encoded = []
    
    for cat in categories:
        encoded.append(category_mean[cat])
    
    return encoded
