struct KNNOOD
    tree::KDTree
    k::Int
    threshold::Float64
end

function fit_knn_ood(Ztrain::AbstractMatrix; k=5, q=0.99)
    tree = KDTree(Ztrain)
    
    # score each training point against its neighbors
    # first neighbor is itself, so use k+1 and skip first
    train_scores = Float64[]
    for i in axes(Ztrain, 2)
        idxs, dists = knn(tree, Ztrain[:, i], k + 1, true)
        push!(train_scores, mean(dists[2:end]))
    end
    # display(plot(train_scores))
    threshold = quantile(train_scores, q)
    return KNNOOD(tree, k, threshold)
end

function KNN_score(model::KNNOOD, z::AbstractVector)
    idxs, dists = knn(model.tree, z, model.k, true)
    return mean(dists)
end