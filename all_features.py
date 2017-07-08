"""
Determine the gender of a person based on Name. This program uses the Random Forest Classifier. 
"""
import collections
from nltk.corpus import names #You won't need this, I'll provide the files
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from string import ascii_uppercase

#Originally from nltk.corpus
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


#Create One-hot Encoding dictionary from element string
def create_one_hot(eles):
	""" Generate the One Hot Encode (OHE) for a given input.

	One-hot comes from originally from electronics - one-hot meaning there's only 1 "hot" or "on" value in this list, while the rest are "cold". OHE is used to convert a non-numeric alphabet and alphabet pairs into a numeric vector. 

	Args
		eles: a list of all possible labels

	Returns
		one_hot: 
			Dictionary of every element as the key and an N dimensional vector with one 1 and N-1 other entries as 0 (hence the name 'ONE-HOT'). N is the size of the input element list eles. 
	"""
	one_hot = {}
	for i, l in enumerate(eles):
		bits = [0]*len(eles);	#Every element in the string/list is assigned 0
		bits[i] = 1;	#Only one bit is set to "ON"
		one_hot[l] = bits 	#Actual assignment is made
	return one_hot

mono_alpha_hot = create_one_hot(ascii_uppercase)
gender_hot = create_one_hot('MF')

#Create Bi-Alphabets
bi_alphabets = [a+b for a in ascii_uppercase for b in ascii_uppercase]
bi_alpha_hot = create_one_hot(bi_alphabets)

"""Alphabet Triplets is not recommended. Comment out to see anyways."""
# #Crete Alphabet Triplets
# tri_alphabets = [a+b+c for a in ascii_uppercase for b in ascii_uppercase for c in ascii_uppercase]
# tri_alpha_hot = create_one_hot(tri_alphabets)

"""Create Feature names"""
feat_names = []
feat_names.extend( ['Starts with '+a for a in  mono_alpha_hot.keys()] )
feat_names.extend( ['2nd Character '+a for a in  mono_alpha_hot.keys()] )
feat_names.extend( ['2nd Character from last '+a for a in  mono_alpha_hot.keys()] )
feat_names.extend( ['Ends with '+a for a in  mono_alpha_hot.keys()] )
feat_names.extend( ['Freqency of '+a for a in list(ascii_uppercase)] )
feat_names.extend( ['Contains '+a for a in list(bi_alphabets)] )

def get_sample(name, gender):
	""" Get the features of an input sample.

	The method takes a training sample as input and computes the following features from every name. The vecor size is given in parentheses:
		- First letter (26)
		- Last Letter  (26)
		- Second Letter (26)
		- Sencond from last Letter (26)
		- Freq alphabet (26)
		- Freq bi-alphabet (26 x 26)
		- Freq tri-alphabet (26 x 26 x 26) -- takes too long

	Args
		name: name of a person
		gender: The label corresponding to the gender. Possible values are Male 'M' or Female 'F'.

	Returns
		tuple:
			features : list of numeric feature values.
			classification : '0' for Male and '1' for Female.

	"""
	features = []
	name = name.strip()

	##First Character
	features.extend( mono_alpha_hot[name[0]] )

	##Second Character
	features.extend( mono_alpha_hot[name[1]] )

	##Second Character from Last
	features.extend( mono_alpha_hot[name[-2]] )

	##Last Character
	features.extend( mono_alpha_hot[name[-1]] )

	##Frequency
	freq = {key : 0 for key in list(ascii_uppercase)} 	#Initialize all keys to 0 for every Alphabet
	updates = dict(collections.Counter(name))	#Get the frequency distribution of characters in 'name' 
	freq.update(updates)	#update the original values of the dictionary

	features.extend( freq.values() ) #Append the list of values

	##bi-alphabet
	freq = {key : 0 for key in list(bi_alphabets)}#Initialize all keys to 0 for every Alphabet Pair
	updates = dict(collections.Counter( zip(name, name[1:]) )) 	#Freq. Distribution of Alphabet pairs in the name in the form (A,B): n
	updates = {(A+B):n for (A,B),n in zip(updates.keys(),updates.values())}	#Convert (A,B) : n format to dictionary of "AB" : n.
	freq.update(updates)

	features.extend( freq.values() ) #Append the list of values

	"""Alphabet triplets take way to long. Here is the code."""
	##tri-alphabet
	# freq = {key : 0 for key in list(tri_alphabets)}
	# updates = dict(collections.Counter( zip(name, name[1:], name[2:]) ))
	# updates = {(A+B+C):n for (A,B,C),n in zip(updates.keys(),updates.values())}
	# freq.update(updates)
	# features.extend( freq.values() )


	#Gender Label 
	if gender == 'M': 
		classification = 0
	else:
		classification = 1

	return (features, classification)

feature_list = [ get_sample(name, gender) for name, gender in all_names]

#Shuffle list to make sure Male And Female are mixed well
random.shuffle(feature_list)

#Split test and train set
train_set = feature_list[:7000]
test_set = feature_list[7000:]

#Conversion to the correct format
X_train, y_train = zip(*train_set) #converts list of 2-field tuples to 2 separate lists
X_test, y_test = zip(*test_set)

classifier = RandomForestClassifier(n_estimators=150, min_samples_split=20)
classifier.fit(X_train, y_train)	#Performs the actual "training" phase

y_pred = []
for i in range(0,len(X_test)):
	y_pred.extend(classifier.predict(np.array(X_test[i]).reshape(1, -1))) #Reshape to avoid deprecation warning


print(accuracy_score(y_test, y_pred))

important_features = sorted(enumerate(classifier.feature_importances_), key=lambda x : x[1], reverse=True)
print ("Most Important Features : ", [(feat_names[idx],prob) for idx, prob in important_features][:20])

