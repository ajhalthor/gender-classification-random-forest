"""
Gender Classification using only the top features from `forest_main.py`.

We only use the top features: 
	- Ends with A
	- Frequency of A
	- Ends with E
	- 2nd to last letter is N

"""
import collections
import numpy as np
from nltk.corpus import names
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import pprint as pp

male_names =  names.words('male.txt')
female_names =  names.words('female.txt')

#Get rid of names with non_alphabetic characters
male_names = filter(str.isalpha, [str(m) for m in male_names]) #Convert unicode array to string array
female_names = filter(str.isalpha, [str(f) for f in female_names])

#Convert Names to same case. 'A' != 'a'
all_names = []
for name in male_names:
	all_names.append( (name.upper(),'M') )

for name in female_names:
	all_names.append( (name.upper(),'F') )

def get_sample(name, gender):
	""" Get the features of an input sample.

	The method takes a training sample as input and computes the following 4 features from every name:
		- Ends with A
		- Frequency of A
		- Ends with E 
		- Second from last character is N

	Args
		name: name of a person
		gender: The label corresponding to the gender. Possible values are Male 'M' or Female 'F'.

	Returns
		tuple:
			features : list of numeric feature values. (4 x 1)
			classification : '0' for Male and '1' for Female.

	"""
	features = []
	name = name.strip()
	
	##Ends with A
	if name[-1] == 'A':
		features.append(1)
	else:
		features.append(0)

	##Ends with 'E'
	if name[-1] == 'E':
		features.append(1)
	else:
		features.append(0)

	#Freq of A
	features.append( name.count('A') )

	##2nd character from end is N
	if name[-2] == 'N':
		features.append(1)
	else:
		features.append(0)

	#Gender Label 
	if gender == 'M': 
		classification = 0
	else:
		classification = 1

	return (features, classification)

feature_list = [ get_sample(name, gender) for name, gender in all_names]
print("Accuracy with top 4 features")
for i in range(10):
    #Shuffle list to make sure Male And Female are mixed well
    random.shuffle(feature_list)

    #Split test and train set
    train_set = feature_list[:7000]
    test_set = feature_list[7000:]

    #Conversion to the correct format
    X_train, y_train = zip(*train_set) #converts list of 2-field tuples to 2 separate lists
    X_test, y_test = zip(*test_set)

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=150, min_samples_split=20)
    classifier.fit(X_train, y_train)	#Performs the actual "training" phase

    y_pred = []
    for j in range(0,len(X_test)):
        y_pred.extend(classifier.predict(np.array(X_test[j]).reshape(1, -1)))

    print("Epoch ", i , " : ", accuracy_score(y_test, y_pred))

