def label_encoding(categories):
    """
    Map each unique category to an integer using a set and a loop.
    """
    # Step 1: get unique categories
    unique_categories = set(categories)
    
    # Step 2: manually map each category to an integer
    category_map = {}
    i = 1   # start with 1 if you want, or 0
    for cat in unique_categories:
        category_map[cat] = i
        i += 1
    
    # Step 3: rebuild the encoded list
    encoded = []
    for cat in categories:
        encoded.append(category_map[cat])
    
    return encoded
