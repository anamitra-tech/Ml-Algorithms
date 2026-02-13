def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    from collections import Counter
    
    sorted_vals = sorted(values)
    freq = Counter(sorted_vals)
    
    rank_map = {}
    i = 0
    n = len(sorted_vals)
    
    while i < n:
        value = sorted_vals[i]
        f = freq[value]
        
        if f == 1:
            rank_map[value] = i + 1
        else:
            start = i + 1
            end = i + f
            rank_map[value] = (start + end) / 2
        
        i += f
    
    return [rank_map[x] for x in values]
