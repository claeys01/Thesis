struct KNNOOD
    tree::KDTree
    k::Int
    threshold::Float64
end

function fit_knn_ood(Ztrain::AbstractMatrix; k=5, q=0.99)
    Znorm = Ztrain ./ norm.(eachcol(Ztrain))'
    tree = KDTree(Znorm)
    
    # score each training point against its neighbors
    # first neighbor is itself, so use k+1 and skip first
    train_scores = Float64[]
    for i in axes(Ztrain, 2)
        # idxs, dists = knn(tree, Ztrain[:, i], k + 1, true)
        idxs, dists = knn(tree, Znorm[:, i], k + 1, true)
        push!(train_scores, mean(dists[2:end]))
    end
    # display(plot(train_scores))
    if q < 1
        threshold = quantile(train_scores, q)
    else
        threshold = quantile(train_scores, 1) * q
    end
    return KNNOOD(tree, k, threshold)
end

function KNN_score(model::KNNOOD, z::AbstractVector)
    znorm = z / norm(z)
    idxs, dists = knn(model.tree, znorm, model.k, true)
    # return mean(dists)
    return dists[end]
end