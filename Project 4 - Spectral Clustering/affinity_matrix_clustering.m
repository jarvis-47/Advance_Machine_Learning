function [clusters, eigValues] = affinity_matrix_clustering(dataset, k, sigma)

    % SPECTRAL CLUSTERING USING AN AFFINITY MATRIX
    
    % Args:
    % Dataset - Vector or matrix of observations/datapoints
    
    % Return:
    % Clustered vector or matrix. eg: [1 1 2 2] means first two data points
    % belong to cluster 1 and last two belong to cluster 2.
    
    A = exp(-pdist2(double(dataset), double(dataset), "squaredeuclidean")/sigma);
    
    % Step 2: Symmetrically normalize the affinity matrix
    D = diag(1 ./ sqrt(sum(A,2)));
    N = D * A * D;
    
    % Step 3: Construct matrix Y of the first k eigenvectors of N
    [eigVectors, eigValues] = eigs(N, k, 'largestabs');
    Y = eigVectors;
    
    % Step 4: Normalize each row of Y to have unit length
    Y = Y ./ sqrt(sum(Y.^2, 2));
    
    % Step 5: Cluster the dataset by running k-means on the rows of Y
    clusters = kmeans(Y, k);
end
