def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    features = []
    
    for x in values:
        row = []
        for d in range(0, degree + 1):
            row.append(x ** d)
        features.append(row)
    
    return features
