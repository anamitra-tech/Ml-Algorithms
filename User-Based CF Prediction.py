def user_based_cf_prediction(ratings, similarities):
    # Filter only users with positive similarity
    numerator = 0.0
    denominator = 0.0
    
    for r, s in zip(ratings, similarities):
        if s > 0:  # only positive similarity
            numerator += r * s
            denominator += s
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator  # weighted average
