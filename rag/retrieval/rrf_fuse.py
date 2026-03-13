def rrf_fuse(list1, list2, k=30):

    scores = {}

    for idx,rank in list1:
        scores[idx] = scores.get(idx,0) + 1/(k+rank)

    for idx,rank in list2:
        scores[idx] = scores.get(idx,0) + 1/(k+rank)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return ranked
