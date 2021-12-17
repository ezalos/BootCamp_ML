from my_linear_regression import MyLinearRegression
import numpy as np

class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)
        self.lambda_ = lambda_
        self.theta_ = np.copy(self.theta)
        self.theta_[0][0] = 0
        
    def l2_sum(self, theta):
        return np.sum(np.square(theta[1:]))
    
    def loss_elem_(self, x, y):
        loss = super().loss_elem_(x, y)
        regularization = (self.lambda_ * self.l2_sum(self.theta))
        ridge_loss = loss + regularization
        return ridge_loss
    
    def loss_(self, y_hat, y):
        # y_hat = self.predict_(x)
        loss = np.sum(np.square(y_hat - y))
        regularization = (self.lambda_ * self.l2_sum(self.theta_))
        ridge_loss = loss + regularization
        return ridge_loss

    def gradient_(self, x, y, y_hat):
        classic_grad = super().gradient_(x, y, y_hat)
        regularization = ((self.lambda_ * self.theta_) / y.shape[0])
        return classic_grad + regularization

    def predict(self, x):
        return self.predict_(x)
    
    def get_params_(self):
        params = {}
        params["theta"] = self.theta
        params["alpha"] = self.alpha
        params["max_iter"] = self.max_iter
        params["lambda_"] = self.lambda_
        return (params)
    
    def set_params_(self, **kwargs):
        for key, value in kwargs.items():
            if (key == "lambda_"):
                self.lambda_ = value
            elif (key == "theta"):
                self.theta = value
            elif (key == "max_iter"):
                self.max_iter = value
            elif (key == "alpha"):
                self.alpha = value

if __name__ == "__main__":
    print("X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])")
    print("Y = np.array([[23.], [48.], [218.]])")
    print("mylr = MyRidge([[1.], [1.], [1.], [1.], [1]], lambda_=1)\n")
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyRidge([[1.], [1.], [1.], [1.], [1]], lambda_=1)

    print(f"{mylr.predict_(X) = }\n")
    print(f"{mylr.loss_(mylr.predict_(X),Y) = }")
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    print(f"{mylr.fit_(X, Y) = }")
    print(f"{mylr.predict_(X) = }\n")
    print(f"{mylr.loss_(mylr.predict_(X),Y) = }")
