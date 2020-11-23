"""  Version 2.3.3  """

import smtplib
import numpy as np
import pandas as pd 
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier, plot_importance
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from matplotlib import style
sns.set(style='ticks', palette='muted')
from subprocess import check_output
pd.options.display.max_colwidth = 1000
pd.options.display.max_rows = 100


df = pd.read_csv('student dataset.csv')     #Dataset used from (https://www.kaggle.com/aljarah/xAPI-Edu-Data)

# Any results you write to the current directory are saved as output.
df.head()
print("\n")
df.describe
print(df.shape)
print("\n")
df.isnull().sum()
len(df)
df.head(4)
df.columns
df['gender'].value_counts()
df['PlaceofBirth'].value_counts()

##########################################################################################################################

def evaluate_data(df):
    # Check for range of unique values for the train data
    for i in range(df.shape[1]):
        vals = np.unique(df.iloc[:, i])
        if len(vals) < 15:
            print (df.columns[i], ': (Categorical) {} unique value(s) - {}'.format(len(vals), vals))
        else:
            if df.iloc[:, i].dtype == object:
                print (df.columns[i], ': (Continuous) range of values of type string {',df.iloc[:, i].unique().size,' values}')
            else:
                print (df.columns[i], ': (Continuous) range of values - ', '[ {} to {}]'.format(df.iloc[:, i].min(), df.iloc[:, i].max()), ' {',df.iloc[:, i].unique().size,' values}')

def columns_with_null(df):
    cnt = 0
    for column in df.columns:
        df_missing = df[df[column].isnull()]
        if df_missing.shape[0] > 0:
            print ("Column " , column, " contain null values / Count = " ,df_missing.shape[0])
            cnt = cnt + 1
    
    if cnt ==0:
        print ("The dataframe does not have 'null' values in any column")

evaluate_data(df)


###########################################################################################################################

malelowlevel=0
malemediumlevel=0
malehighlevel=0
femalelowlevel=0
femalemediumlevel=0
femalehighlevel=0
dosum=0

topMLowerlevel =0
topMMiddleSchool=0
topMHighSchool=0
topFLowerlevel =0
topFMiddleSchool=0
topFHighSchool=0

topperformer = []
index =2

for g,sid,raisedhands,VisITedResources,AnnouncementsView,Discussion in zip(df.gender,df.StageID,df.raisedhands,df.VisITedResources,df.AnnouncementsView,df.Discussion):
	if(g == 'M' and sid == 'lowerlevel'):
		malelowlevel=malelowlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMLowerlevel = topMLowerlevel +1
			topperformer.append(index)
		index=index+1

			

	if(g == 'M' and sid == 'MiddleSchool'):
		malemediumlevel=malemediumlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMMiddleSchool = topMMiddleSchool +1
			topperformer.append(index)
		index=index+1

	if(g == 'M' and sid == 'HighSchool'):
		malehighlevel=malehighlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMHighSchool = topMHighSchool +1
			topperformer.append(index)
		index=index+1


	if(g == 'F' and sid == 'lowerlevel'):
		femalelowlevel=femalelowlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFLowerlevel = topFLowerlevel +1
			topperformer.append(index)
		index=index+1

	if(g == 'F' and sid == 'MiddleSchool'):
		femalemediumlevel=femalemediumlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFMiddleSchool = topFMiddleSchool +1
			topperformer.append(index)
		index=index+1

	if(g == 'F' and sid == 'HighSchool'):
		femalehighlevel=femalehighlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFHighSchool = topFHighSchool +1
			topperformer.append(index)
		index=index+1

print("\n")	
print("These are the active students:")
print("\n")
print(topperformer)
print("\n")
print("Further data is classified into how much active students are in each category")
print("\n")
print("Analyzes: Male with lowlevel:","there are top",topMLowerlevel,"Performer Students out of",malelowlevel)
print("Analyzes: Male with Mediumlevel:","there are top",topMMiddleSchool,"Performer Students out of",malemediumlevel)
print("Analyzes: Male with Highlevel:","there are top",topMHighSchool,"Performer Students out of",malehighlevel)
print("Analyzes: Female with lowlevel:","there are top",topFLowerlevel,"Performer Students out of",femalelowlevel)
print("Analyzes: female with Mediumlevel:","there are top",topFMiddleSchool,"Performer Students out of",femalemediumlevel)
print("Analyzes: female with Highlevel:","there are top",topFHighSchool,"Performer Students out of",femalehighlevel)
print("\n")

###########################################################################################################################
sns.pairplot(df,hue='Class') 
plt.show()

categorical_features = (df.select_dtypes(include=['object']).columns.values)
categorical_features
mod_df = df 
numerical_features = df.select_dtypes(include = ['float64', 'int64']).columns.values
numerical_features

pivot = pd.pivot_table(df,
            values = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'],
            index = ['gender', 'NationalITy', 'PlaceofBirth'], 
                       columns= ['ParentschoolSatisfaction'],
                       aggfunc=[np.mean, np.std], 
                       margins=True)

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)
plt.subplots(figsize = (30, 20))
sns.heatmap(pivot,linewidths=0.2,square=True )
plt.show()

def heat_map(corrs_mat):
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(20, 20))
    mask = np.zeros_like(corrs_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True 
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrs_mat, mask=mask, cmap=cmap, ax=ax)
    plt.show()

variable_correlations = df.corr()

heat_map(variable_correlations)

###########################################################################################################################
###########################################################################################################################

## testing 
#breakdown by class
sns.countplot(x="Topic", data=df, palette="muted");
plt.show()

df['Failed'] = np.where(df['Class']=='L',1,0)
sns.factorplot('Topic','Failed',data=df,size=9) # '' not showing ''
plt.show()

pd.crosstab(df['Class'],df['Topic'])
sns.countplot(x='Class',data=df,palette='muted')
plt.show()

df.Class.value_counts()
sns.countplot(x='ParentschoolSatisfaction',data = df, hue='Class',palette='muted')
plt.show()

Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df)
Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df, color=".15")
plt.show()

ax = sns.boxplot(x="Class", y="Discussion", data=df)
ax = sns.swarmplot(x="Class", y="Discussion", data=df, color=".25")
plt.show()

Vis_res = sns.boxplot(x="Class", y="VisITedResources", data=df)
Vis_res = sns.swarmplot(x="Class", y="VisITedResources", data=df, color=".25")
plt.show()

Anounce_bp = sns.boxplot(x="Class", y="AnnouncementsView", data=df)
Anounce_bp = sns.swarmplot(x="Class", y="AnnouncementsView", data=df, color=".25")
plt.show() 

nationality = sns.countplot(x = 'PlaceofBirth', data=df, palette='muted')
nationality.set(xlabel='PlaceofBirth',ylabel='count', label= "Students Birth Place")
plt.setp(nationality.get_xticklabels(), rotation=90)
plt.show()

pd.crosstab(df['Class'],df['Topic'])

sns.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='muted')
plt.show()

P_Satis = sns.countplot(x="ParentschoolSatisfaction",data=df,linewidth=2,edgecolor=sns.color_palette("muted"))
plt.show()

plot = sns.countplot(x='Class', hue='Relation', data=df, order=['L', 'M', 'H'], palette='muted')
plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')
plt.show()

sns.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='muted')
plt.show()

mod_df=df
gender_map = {'M':1, 
              'F':2}

NationalITy_map = {  'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}
PlaceofBirth_map =  {'Iran': 1,
                     'SaudiArabia': 2,
                     'USA': 3,
                     'Egypt': 4,
                     'Lybia': 5,
                     'lebanon': 6,
                     'Morocco': 7,
                     'Jordan': 8,
                     'Palestine': 9,
                     'Syria': 10,
                     'Tunis': 11,
                     'KW': 12,
                     'KuwaIT': 12,
                     'Iraq': 13,
                     'venzuela': 14}

StageID_map = {'HighSchool':1, 
               'lowerlevel':2, 
               'MiddleSchool':3}

GradeID_map =   {'G-02':2,
                 'G-08':8,
                 'G-09':9,
                 'G-04':4,
                 'G-05':5,
                 'G-06':6,
                 'G-07':7,
                 'G-12':12,
                 'G-11':11,
                 'G-10':10}

SectionID_map = {'A':1, 
                 'C':2, 
                 'B':3}

Topic_map  =    {'Biology' : 1,
                 'Geology' : 2,
                 'Quran' : 3,
                 'Science' : 4,
                 'Spanish' : 5,
                 'IT' : 6,
                 'French' : 7,
                 'English' :8,
                 'Arabic' :9,
                 'Chemistry' :10,
                 'Math' :11,
                 'History' : 12}
Semester_map = {'S':1, 
                'F':2}

Relation_map = {'Mum':2, 
                'Father':1} 

ParentAnsweringSurvey_map = {'Yes':1,
                             'No':0}

ParentschoolSatisfaction_map = {'Bad':0,
                                'Good':1}

StudentAbsenceDays_map = {'Under-7':0,
                          'Above-7':1}

Class_map = {'H':10,
             'M':5,
             'L':2}

mod_df.gender  = mod_df.gender.map(gender_map)
mod_df.NationalITy     = mod_df.NationalITy.map(NationalITy_map)
mod_df.PlaceofBirth     = mod_df.PlaceofBirth.map(PlaceofBirth_map)
mod_df.StageID       = mod_df.StageID.map(StageID_map)
mod_df.GradeID = mod_df.GradeID.map(GradeID_map)
mod_df.SectionID    = mod_df.SectionID.map(SectionID_map)
mod_df.Topic     = mod_df.Topic.map(Topic_map)
mod_df.Semester   = mod_df.Semester.map(Semester_map)
mod_df.Relation   = mod_df.Relation.map(Relation_map)
mod_df.ParentAnsweringSurvey   = mod_df.ParentAnsweringSurvey.map(ParentAnsweringSurvey_map)
mod_df.ParentschoolSatisfaction   = mod_df.ParentschoolSatisfaction.map(ParentschoolSatisfaction_map)
mod_df.StudentAbsenceDays   = mod_df.StudentAbsenceDays.map(StudentAbsenceDays_map)
mod_df.Class  = mod_df.Class.map(Class_map)

categorical_features = (mod_df.select_dtypes(include=['object']).columns.values)
categorical_features

mod_df_variable_correlations = mod_df.corr()
#variable_correlations
heat_map(mod_df_variable_correlations)


df_copy = pd.get_dummies(df)

df1 = df_copy
y = np.asarray(df1['ParentschoolSatisfaction'], dtype="|S6")
df1 = df1.drop(['ParentschoolSatisfaction'],axis=1)
X = df1.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)

radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)
indices = np.argsort(radm.feature_importances_)[::-1]

# Print the feature ranking
print("\n")
print('Feature ranking:')

for f in range(df1.shape[1]):
    print('%d. feature %d %s (%f)' % (f+1 , 
                                      indices[f], 
                                      df1.columns[indices[f]], 
                                      radm.feature_importances_[indices[f]]))

# --------------------------------------------------------------------------
df = pd.read_csv('student dataset.csv')

df.groupby('Topic').median()

df['AbsBoolean'] = df['StudentAbsenceDays']
df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)
df['AbsBoolean'].groupby(df['Topic']).mean()
df[9:13].describe()

df['TotalQ'] = df['Class']
df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0

continuous_subset = df.ix[:,9:13]
continuous_subset['gender'] = np.where(df['gender']=='M',1,0)
continuous_subset['Parent'] = np.where(df['Relation']=='Father',1,0)

X = np.array(continuous_subset).astype('float64')
y = np.array(df['TotalQ'])
X.shape

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=20)


sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


ppn = Perceptron(n_iter=50, eta0=0.01, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("\n")
print("----------------------------Perceptron-------------------------------")
print(classification_report(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("\n")



X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='linear', C=2.0, random_state=0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print("-------------------------------SVC_linear----------------------------")
print(classification_report(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("\n")


X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.28, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print("-------------------------------SVC_rbf------------------------------")
print(classification_report(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("\n")

svm = SVC(kernel='poly', random_state=0, degree=3, gamma=2, C=0.01)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print("-------------------------------SVC_poly------------------------------")
print(classification_report(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("\n")
#---------------------------------------------------------------------------
continuous_subset['Absences'] = df['AbsBoolean'] 
X = np.array(continuous_subset).astype('float64')
y = np.array(df['TotalQ'])
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=0)
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print("--------------------------SC---------------------------------")
print(classification_report(y_test, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print("\n")

df.loc[(df['raisedhands']==2) & (df['VisITedResources']==9) & (df['AnnouncementsView']==7)]

clf = MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=0)

sc = StandardScaler()
sc.fit(X)

clf = MLPClassifier(solver='lbfgs',alpha=.1,random_state=0)
clf.fit(X,y)
scores=cross_val_score(clf,X,y,cv=10)

print('Mean Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
print("\n")
#####################################################################################################################################################################################################

df.dtypes
Features = df.drop('Class',axis=1)
Target = df['Class']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col]) 


''' ----- Logistic Regression ----- '''

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=30)

Logit_Model = LogisticRegression()

Logit_Model.fit(X_train,y_train)
Prediction = Logit_Model.predict(X_test)
Score = accuracy_score(y_test,Prediction)
Report = classification_report(y_test,Prediction)
print('-----------------------Logistic Regression-----------------------')
print("\n")
print('Prediction: \n', Prediction)
print("\n")
print(Report)
print('Accuracy: ', Score)
print("\n")


''' ----- XGBoost ----- '''

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=0)

xgb = XGBClassifier(max_depth=100, learning_rate=0.01, n_estimators=1000,seed=10)

xgb_pred = xgb.fit(X_train, y_train).predict(X_test)
print('----------------------------XGBoost-------------------------------')
print("\n")
print('Prediction: \n', xgb_pred)
print("\n")
print(classification_report(y_test,xgb_pred))
print('Accuracy: ', accuracy_score(y_test,xgb_pred))