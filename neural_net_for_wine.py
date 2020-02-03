#Tutorial from https://www.datacamp.com/community/tutorials/deep-learning-python
# Import pandas, numpy and matplotlib for visualization
import os
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def collect_info_from_sheets():
	"""
	None - > Dataframe, Dataframe
	This function is in charge of collecting information on the two types of wine from the csv files and 
	returning two dataframes
	"""
	# Read in white wine data 
	white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
	# Read in red wine data 
	red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

	return white, red

def check_collected_info(white, red):
	"""
	Dataframe, Dataframe -> None 
	This function is in charge of checking/displaying information on the collected data frames
	"""
	#print info of wine categories that have been obtained
	print(white.info())
	print(red.info())

	#further check the collected data
	# First rows of `red` 
	red.head()
	# Last rows of `white`
	white.tail()
	# Take a sample of 5 rows of `red`
	red.sample(5)
	# Describe `white`
	white.describe()
	# Double check for null values in `red`
	pd.isnull(red)
	return

def visualize_data(white, red):
	fig, ax = plt.subplots(1, 2)

	ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
	ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

	fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
	ax[0].set_ylim([0, 1000])
	ax[0].set_xlabel("Alcohol in % Vol")
	ax[0].set_ylabel("Frequency")
	ax[1].set_xlabel("Alcohol in % Vol")
	ax[1].set_ylabel("Frequency")
	#ax[0].legend(loc='best')
	#ax[1].legend(loc='best')
	fig.suptitle("Distribution of Alcohol in % Vol")

	plt.show()
	return 

def compute_histograms(red, white):
	"""
	DataFrame, DataFrame -> None
	This function is in charge of computing a histogram for each of the DataFrames collected
	"""
	print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
	print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
	return

def check_wine_quality_and_display(white, red):
	"""
	DataFrame, DataFrame -> None
	This function is in charge of displaying the quality of each wine in terms of Sulphates
	"""
	fig, ax = plt.subplots(1, 2, figsize=(8, 4))

	ax[0].scatter(red['quality'], red["sulphates"], color="red")
	ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

	ax[0].set_title("Red Wine")
	ax[1].set_title("White Wine")
	ax[0].set_xlabel("Quality")
	ax[1].set_xlabel("Quality")
	ax[0].set_ylabel("Sulphates")
	ax[1].set_ylabel("Sulphates")
	ax[0].set_xlim([0,10])
	ax[1].set_xlim([0,10])
	ax[0].set_ylim([0,2.5])
	ax[1].set_ylim([0,2.5])
	fig.subplots_adjust(wspace=0.5)
	fig.suptitle("Wine Quality by Amount of Sulphates")

	plt.show()
	return 

def check_acidity(red, white):
	"""
	DataFrame, DataFrame -> None
	This function is in charge of displaying the acidity of each wine. This is doing so
	by using scatter plots
	"""
	np.random.seed(570)
	redlabels = np.unique(red['quality'])
	whitelabels = np.unique(white['quality'])
	fig, ax = plt.subplots(1, 2, figsize=(8, 4))
	redcolors = np.random.rand(6,4)
	whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

	for i in range(len(redcolors)):
	    redy = red['alcohol'][red.quality == redlabels[i]]
	    redx = red['volatile acidity'][red.quality == redlabels[i]]
	    ax[0].scatter(redx, redy, c=redcolors[i])

	for i in range(len(whitecolors)):
	    whitey = white['alcohol'][white.quality == whitelabels[i]]
	    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
	    ax[1].scatter(whitex, whitey, c=whitecolors[i])
	    
	ax[0].set_title("Red Wine")
	ax[1].set_title("White Wine")
	ax[0].set_xlim([0,1.7])
	ax[1].set_xlim([0,1.7])
	ax[0].set_ylim([5,15.5])
	ax[1].set_ylim([5,15.5])
	ax[0].set_xlabel("Volatile Acidity")
	ax[0].set_ylabel("Alcohol")
	ax[1].set_xlabel("Volatile Acidity")
	ax[1].set_ylabel("Alcohol") 
	#ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
	ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
	#fig.suptitle("Alcohol - Volatile Acidity")
	fig.subplots_adjust(top=0.85, wspace=0.7)

	plt.show()
	return 

def pre_process_data(white, red):
	"""
	DataFrame, DataFrame -> list
	This preprocess's the data so it can be distinguished later on
	"""
	# Add `type` column to `red` with value 1
	red['type'] = 1

	# Add `type` column to `white` with value 0
	white['type'] = 0

	# Append `white` to `red`
	wines = red.append(white, ignore_index=True)
	return wines


def plot_coorelation_matrix(wines):
	"""
	list -> None
	This plots a coorelation matrix of the collected wine samples
	"""
	corr = wines.corr()
	sns.heatmap(corr, 
	            xticklabels=corr.columns.values,
	            yticklabels=corr.columns.values)
	plt.show()
	return

def train_and_test(wines):
	# Specify the data 
	X=wines.ix[:,0:11]

	# Specify the target labels and flatten the array
	y = np.ravel(wines.type)

	# Split the data up in train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	#Now standardize the data from train_test
	#define the scaler
	scaler = StandardScaler().fit(X_train)

	# Scale the train set
	X_train = scaler.transform(X_train)

	# Scale the test set
	X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, scaler

def init_neural_net():
	# Initialize the constructor
	model = Sequential()

	#more units more leanring of complex representations, also expensive
	# Add an input layer
	#12 is the input arrays of shape 12. 12 hidden unites for the input layer
	model.add(Dense(12, activation='relu', input_shape=(11,)))

	# Add one hidden layer 
	#The intermediate layer also uses the relu activation function. The output of this layer will be arrays of shape (*,8).
	model.add(Dense(8, activation='relu'))

	# Add an output layer 
	#Dense layer of size one. This means the output will either be a zero or one indicating red or white wine
	model.add(Dense(1, activation='sigmoid'))

	return model

def model_info(model):
	model.output_shape

	# Model summary
	model.summary()

	# Model config
	model.get_config()

	# List all weight tensors 
	model.get_weights()
	return

def compile_and_fit(model, X_train, y_train):
	#OPTIMIZER AND LOSS ARE TWO ARGUMENTS THAT ARE REQUIRED
	# Adam is the optimization algorithm being used
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
	return

def predict_model(model, y_test, X_test):
	y_pred = model.predict(X_test)
	y_pred[:5]
	y_test[:5]
	return y_pred

def evaluate_model(model, X_test, y_test):
	#visually compare the predictions with the actual test labels to determine actual performance.
	score = model.evaluate(X_test, y_test, verbose=1)
	#a list of two things, loss and accuracy. The data is still imbalanced here though. This is because there is more white wine in the sample
	print("Evaluation Score: {}".format(score))
	return

def create_confusion_matrix(y_test, y_pred):
	#confusion_matrix(y_test, y_pred) # The breakdown of predictions into a table showing correct predictions and the types of incorrect predictions made.
	
	precision_score(y_test, y_pred) #a measure of a classifiers exactness, the higher the precision the more accurate the classifier is
	
	recall_score(y_test, y_pred) # is a measure of the classifiers completeness. The higher the more cases that are coverd
	
	f1_score(y_test, y_pred) # is a measure of the weighted average of the precision and recall
	
	cohen_kappa_score(y_test, y_pred) #is the calssification accuracy normalized by the imbalance of the classes in the data
	return

def pre_process_data_round_2(wines):
	#Isolate target labels
	y = wines.quality
	#Isolate data
	X = wines.drop('quality', axis=1)
	#Also must perform standard scaling again 
	#this is because there was alot of differences
	#in some of the values in red and white wine
	X = StandardScaler().fit_transform(X)
	return X, y

def create_model_neural_net_structure():

	#create the N.N model using the sequential function
	model = Sequential()
	#add the dense layer with the input dimension of 12
	#This is the input layer with 64 hidden units
	model.add(Dense(128, input_dim=12, activation = 'relu'))
	model.add(Dense(128, activation='relu'))
	#add a dense layer
	#This is a typical setup for scalar regression, 
	# where you are trying to predict a single continuous value
	model.add(Dense(1))
	return model

def compile_and_fit_round_2(model, X, Y):
	seed = 7
	np.random.seed(seed)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
	for train, test in kfold.split(X, Y):
		model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
		model.fit(X[train], Y[train], epochs=10, verbose=1)
		mse_value, mae_value = model.evaluate(X[test], Y[test], verbose=0)
	y_pred = model.predict(X[test])
	print(mse_value, mae_value)
	print(r2_score(Y[test], y_pred))
	return

def main():
	""" 
	None -> None
	This is the function that ties the program together
	"""
	white, red = collect_info_from_sheets()
	check_collected_info(white, red)
	visualize_data(white, red)
	compute_histograms(red, white)
	check_wine_quality_and_display(white, red)
	check_acidity(red, white)
	wines = pre_process_data(white, red)
	#plot_coorelation_matrix(wines)
	X_train, X_test, y_train, y_test, scaler = train_and_test(wines)
	model = init_neural_net()
	model_info(model)
	compile_and_fit(model, X_train, y_train)
	y_pred = predict_model(model, y_test, X_test)
	evaluate_model(model, X_test, y_test)
	#create_confusion_matrix(y_test, y_pred)
	x, y = pre_process_data_round_2(wines)
	model_2 = create_model_neural_net_structure()
	compile_and_fit_round_2(model_2, x, y)

	return

if __name__ == '__main__':
	main()