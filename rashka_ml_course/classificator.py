import numpy as np 

class Perceptron:
    """
    Classificator based on perceptron 
    """
    
    eta: float  # Training tempo
    n_iter: int # Number of training iterations
        
   
    w_ : np.ndarray # Function weights
    erros_ : list   # Number of misclassifications
        
    def __init__(self, eta=0.01, n_iter=10) -> None:
        self.eta = eta 
        self.n_iter = n_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ 
        X: training vectors, X = [n_samples, n_features] 
        y: target values, Y = [n_samples]
        
        Return self object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0 
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi 
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X: np.ndarray) -> np.ndarray:
        """ 
        Calculate net input function
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
