import numpy as np
class KMeansScratch:
    def __init__(self, k=3, max_iters=300, tol=1e-4):
        self.k=k; self.max_iters=max_iters; self.tol=tol
    def fit(self,X):
        np.random.seed(42)
        idx=np.random.choice(len(X),self.k,replace=False)
        self.centroids=X[idx]
        for _ in range(self.max_iters):
            d=np.linalg.norm(X[:,None]-self.centroids,axis=2)
            labels=np.argmin(d,axis=1)
            new=np.array([X[labels==i].mean(axis=0) for i in range(self.k)])
            if np.linalg.norm(self.centroids-new)<self.tol: break
            self.centroids=new
        self.labels_=labels
        self.inertia_=np.sum((X-self.centroids[labels])**2)
        return self
