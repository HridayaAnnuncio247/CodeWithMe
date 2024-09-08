import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Simple_Linear_Regression:
	def __init__(self, df):
		"""
		pandas DataFrame df: The data in which 1st column is the predictor and 2nd column is the predictand 
		"""
		self.X = df.iloc[:, :1].values #nd arrays nX1
		self.Y = df.iloc[:, 1:2].values #nd array nX1
		self.n = len(self.X)
		self.m = 0
		self.c = 0
		self.sum_X = 0
		self.sum_Y = 0
		self.mean_Y = 0
		self.mean_X =0
		self.Y_pred = None 
		self.SSR= None
		self.SST = None
		self.R2 = None
		#print(self.X)
		#print(self.Y)

		#data: y, x

		#why use numpy arrays
	def calc_sum(self):
		"""
		This function finds the column sums of the predictor(X) colun and predictand(Y) column)
		"""
		self.sum_X = np.sum(self.X).item()
		self.sum_Y = np.sum(self.Y).item()
		
		

	def calc_mean(self):
		"""
		This function finds the means of the predictor(X) colun and predictand(Y) column)

		"""
		self.mean_X = self.sum_X/self.n
		self.mean_Y = self.sum_Y/self.n

		
	def find_m_and_c(self):
		"""
		This function uses X and Y to find the best values for m and c (the coefficients)
		for the equation y = mx + c
		:return: NIL
		"""
		XY = self.X * self.Y
		sum_XY = np.sum(XY).item()
		X2 = self.X *self.X
		sum_X2 = np.sum(X2).item()

		# m = (sum of xi,yi - (sum of x * sum of y)/n)/ (sum of xi^2 - (sum of xi the whole square/n))
		self.m = (sum_XY - ((self.sum_X*self.sum_Y)/self.n)) / (sum_X2 - ((self.sum_X)**2/self.n))

		# c= mean of y - m*mean of x
		self.c = self.mean_Y - self.m*self.mean_X

		


	#find all predicted ys
	def predicted_y(self):
		"""
		This functions sets the numpy array that stores all the predicted Y values.
		The prediction is made by substituting our best m and c along with the X values in the equation y = mx + c. 
		"""
		self.Y_pred = self.m*self.X + self.c
		


	#finding R2 value
	def calc_SSR(self):
		"""
		This function calculates the Sum of Squares Regression = sum of ((Ypredicted - Ymean)^2).
		"""
		self.SSR = np.sum((self.Y_pred-self.mean_Y)**2).item()


	def calc_SST(self):
		"""
		This function calculates the Sum of Squares Total = sum of ((Yactual - Ymean)^2).
		"""
		self.SST = np.sum((self.Y - self.mean_Y)**2).item()
		

	def R_squared(self):
		"""
		This function calculates the value of R-squared(R2). R2 = Sum of Squares Regression/Sum of Squares Total.
		"""
		self.R2 = self.SSR/self.SST


	def plot_data_and_line(self):
		"""
		This functions shows how the regression line looks with respect to all the data.

		"""
		
		plt.scatter(self.X,self.Y)

		# naming the x axis
		plt.xlabel('x = budget for advertising ')
		# naming the y axis
		plt.ylabel('sales')

		# giving a title to my graph
		plt.title('How are the TV sales dependent on the marketing')

		# function to show the plot
		plt.plot(self.X, self.Y_pred, color = "red")
		plt.show()



if __name__ == "__main__":


  # Importing the dataset
  dataset = pd.read_csv('tvmarketing.csv')
  Model = Simple_Linear_Regression(dataset)
  Model.calc_sum()
  Model.calc_mean()
  Model.find_m_and_c()
  Model.predicted_y()
  Model.calc_SSR()
  Model.calc_SST()
  Model.R_squared()
  Model.plot_data_and_line()
  print("R-Squared value", Model.R2)