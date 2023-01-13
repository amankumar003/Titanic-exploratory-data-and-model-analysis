# %% [markdown]
# ## imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from time import time

init_time = time()

# %% [markdown]
# ### Constants

# %%
SURVIVAL_LABEL = ["Survived", "Deceased"]

PASSENGERID = "PassengerId"
SURVIVED    = "Survived"
PCLASS      = "Pclass"
NAME        = "Name"
SEX         = "Sex"
AGE         = "Age"
SIBSP       = "SibSp"
PARCH       = "Parch"
TICKET      = "Ticket"
FARE        = "Fare"
CABIN       = "Cabin"
EMBARKED    = "Embarked"

AGE_RANGE       = "Age_Range"
GENDER_AGE_CAT  = "Gender_Age_Cat"
IS_ALONE        = "Is_Alone"
AGE_PREDICT     = "Age_Predict"

FEATURES_1          = [SURVIVED ,PCLASS ,SEX ,AGE ,SIBSP ,PARCH ,FARE ,EMBARKED]
FEATURES_2          = [GENDER_AGE_CAT, IS_ALONE, AGE_PREDICT]
FEATURES_USELESS    = [PASSENGERID, TICKET, CABIN]

# %% [markdown]
# # Datasets

# %% [markdown]
# ### Data Importing

# %%
test_data_path = "data/test.csv"
train_data_path = "data/train.csv"
submission_data_path = "data/gender_submission.csv"

# %%
df_train = pd.read_csv(train_data_path)
df_train.head()

# %%
df_test = pd.read_csv(test_data_path)
df_test.head()

# %%
datasets = [df_train, df_test]

# %%
df_full = pd.concat(datasets).reset_index(drop=True)

m = df_full.shape[0]
m_test  = df_test.shape[0]
m_train = df_train.shape[0]
print(m, m_train, m_test)
df_full

# %%
def train_test_devide(df_: pd.DataFrame) -> tuple[pd.DataFrame]:
    df_train = df_[(df_full.Survived.notna())]
    df_test = df_[(df_full.Survived.isna())]
    return df_train, df_test



# %%
df_full.nunique()

# %%
df_full.info()

# %% [markdown]
# #### New Data Features

# %%
df_full[IS_ALONE] = ((df_full.Parch + df_full.SibSp) == 0).astype(int)

df_full[IS_ALONE].value_counts()

# %% [markdown]
# ### Predicting Using Weak Model

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

_d_ = df_train[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked", "SibSp", "Parch"]].copy()
_d_ = _d_.dropna()

_encoder_ = LabelEncoder()
_d_.iloc[:, 2] = _encoder_.fit_transform(_d_.iloc[:, 2].values)
_d_.iloc[:, 5] = _encoder_.fit_transform(_d_.iloc[:, 5].values)

_s_scaler_ = StandardScaler()
_d_[["Age", "Fare"]] = _s_scaler_.fit_transform(_d_[["Age", "Fare"]])

_x_ = _d_.iloc[:, 1:].values
_y_ = _d_.Survived.values

_weak_model_ = LogisticRegression().fit(_x_, _y_)

_coef_ = _weak_model_.coef_.round(4).tolist()

sorted(
    list(zip(_d_.columns[1:], _coef_[0][1:]))
    , key=lambda tup: abs(tup[1]), reverse=True)

# %% [markdown]
# Here It can be been that which labels had big role so lets focus on them first...

# %% [markdown]
# ### Dataset Visualization

# %%
# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={
    'figure.figsize':(10,6), 
    'axes.facecolor': '#eee',
    'axes.grid': True,
    'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'},font_scale=1)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)

# %% [markdown]
# #### Survival Status / Various Factors

# %% [markdown]
# ##### Survived vs Deceased

# %%
survival_ratio = df_full.Survived.value_counts(normalize=True)
survival_ratio.plot.barh(color=["C1", "C0"],)

plt.yticks((1, 0), labels=SURVIVAL_LABEL)
plt.title("Survived vs Deceased")
plt.show()

# %%
# sns.set(rc={'figure.figsize':(22,10)})

# ax = sns.countplot(y="answer", hue="sex", data=df)

# # percentage of bars
# for i in ax.patches:
#     # get_width pulls left or right; get_y pushes up or down
#     ax.text(i.get_width()+.12, i.get_y()+.3, \
#             '%' + str(round((i.get_width()/total)*100, 1)), fontsize=15,
#             color='dimgrey')
    
# ax.set_ylabel('Answers',fontsize=20)
# ax.set_xlabel('Count',fontsize=20)
# ax.tick_params(axis='x', which='major', labelsize=20)
# ax.tick_params(axis='y', which='major', labelsize=20)

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
#           prop={'size': 14})


# %% [markdown]
# ##### Survival / Sex (Male vs Female)

# %%
sex_survival_dist_chart = sns.countplot(x=SEX, hue=SURVIVED, data=df_full, palette=["C1", "C0"])
sex_survival_dist_chart.set(ylabel="Percent")

plt.title("Male vs Female Survival")
plt.legend(SURVIVAL_LABEL[::-1])
plt.show()

# %%
sex_survived_dist    = df_full[df_full.Survived==1].Sex.value_counts().sort_index()
sex_deceased_dist    = df_full[df_full.Survived==0].Sex.value_counts().sort_index()
sex_survival_dist    = pd.DataFrame([sex_survived_dist, sex_deceased_dist], index=SURVIVAL_LABEL).T
sex_survival_ratio   = sex_survival_dist.apply(lambda x: x/x.sum(), axis=1)

sex_survival_ratio.plot.bar()
plt.title("Survival Ratio / Sex")
plt.show()

# %% [markdown]
# Here we can see that Sex Really affected the Chance of Survival.<br>
# The Bars are almost inverse of each other.
# 
# * Female - Lucky
# * Male - Unlucky

# %% [markdown]
# ##### Survival / Pclass Difference

# %%
pclass_survived_dist    = df_full[df_full.Survived==1].Pclass.value_counts().sort_index()
pclass_deceased_dist    = df_full[df_full.Survived==0].Pclass.value_counts().sort_index()
pclass_survival_dist    = pd.DataFrame([pclass_survived_dist, pclass_deceased_dist], index=SURVIVAL_LABEL).T
pclass_survival_ratio   = pclass_survival_dist.apply(lambda x: x/x.sum(), axis=1)

pclass_survival_ratio.plot.bar()
plt.title("Survival Ratio / Class")
plt.show()

# %% [markdown]
# Here we can see that People with **higher Pclass** had more chance of survival than the people with **Lower Pclass** .<br>
# Luck-> 1>2>3
# 
# * Pclass 1 - >60% Survival Change
# * Pclass 2 - <50% Survival Change
# * Pclass 3 - =25% Survival Change

# %% [markdown]
# ##### Survival / Age Group

# %%
sns.displot(df_full.Age, bins=30)
plt.title("Age Distribution")
plt.show()

# %%
age_bins = np.arange(0, 81, 10)
Age_Range_df = pd.cut(df_full.Age, bins=age_bins, include_lowest=True)

age_grp_survived_dist    = Age_Range_df[df_full.Survived==1].value_counts().sort_index()
age_grp_deceased_dist    = Age_Range_df[df_full.Survived==0].value_counts().sort_index()
age_grp_survival_dist    = pd.DataFrame([age_grp_survived_dist, age_grp_deceased_dist], index=SURVIVAL_LABEL).T
age_grp_survival_ratio   = age_grp_survival_dist.apply(lambda x: x/x.sum(), axis=1)

age_grp_survival_ratio.plot.bar()
plt.title("Survival Ratio / Age-Group")
plt.show()

# %% [markdown]
# Here we can see that the Chance of Survival is different in different Age-Groups.<br>
# The Bars are mixed, but we can see that - 
# * 0-10 yrs agr group has the Highest chance of survival.
# * 70-80 yrs agr group has the Lowest chance of survival.
# * The Chance of Survial Varies from 60% - 20% depending upon the Age Group...
# 
# Thus The Age of the Passenger is important.

# %% [markdown]
# ##### Survival / Sibbling & Spouse Number 

# %%
sibsp_survived_dist    = df_full[df_full.Survived==1].SibSp.value_counts().sort_index()
sibsp_deceased_dist    = df_full[df_full.Survived==0].SibSp.value_counts().sort_index()
sibsp_survival_dist    = pd.DataFrame([sibsp_survived_dist, sibsp_deceased_dist], index=SURVIVAL_LABEL).T
sibsp_survival_ratio   = sibsp_survival_dist.apply(lambda x: x/x.sum(), axis=1)

sibsp_survival_ratio.plot.bar()
plt.title("Survival Ratio / Sibbling & Spouse number ")
plt.show()

# %% [markdown]
# Here we can see that the Chance of Survival is increased if the passengers is traveling with 1-2 of their Sibling/Spouse then redused after that.<br>
# 
# The Bars follow - 
# * mid-up-mid-down pattern is followed.
# * Having >2 Sibling/Spouse slump the Chance of Survival.
# * The Chance of Survial is best with 1-2 Sibling/Spouse.
# 

# %% [markdown]
# ##### Survival / Parents & Children Number 

# %%
parch_survived_dist    = df_full[df_full.Survived==1].Parch.value_counts().sort_index()
parch_deceased_dist    = df_full[df_full.Survived==0].Parch.value_counts().sort_index()
parch_survival_dist    = pd.DataFrame([parch_survived_dist, parch_deceased_dist], index=SURVIVAL_LABEL).T
parch_survival_ratio   = parch_survival_dist.apply(lambda x: x/x.sum(), axis=1)

parch_survival_ratio.plot.bar()
plt.title("Survival Ratio / Parents & Children Number ")
plt.show()

# %% [markdown]
# Similar to SibSp, here we can see that the Chance of Survival is increased if the passengers is traveling with 1-3 of their Parents/Children then redused after that.<br>
# 
# The Bars follow - 
# * Pattern similar to SibSp
# * Having >3 Parents/Children slump the Chance of Survival.
# * The Chance of Survial is best with 1-3 Parents/Children.
# 

# %% [markdown]
# ##### Survival / Family Members Number 

# %%
fam_num_df = df_full.Parch + df_full.SibSp

fam_num_survived_dist    = fam_num_df[df_full.Survived==1].value_counts().sort_index()
fam_num_deceased_dist    = fam_num_df[df_full.Survived==0].value_counts().sort_index()
fam_num_survival_dist    = pd.DataFrame([fam_num_survived_dist, fam_num_deceased_dist], index=SURVIVAL_LABEL).T
fam_num_survival_ratio   = fam_num_survival_dist.apply(lambda x: x/x.sum(), axis=1)

fam_num_survival_ratio.plot.bar()
plt.title("Survival Ratio / Family Members Number ")
plt.show()

# %% [markdown]
# When we add the number of family members we get a really intresting pattern.<br>
# 
# The Bars follow - 
# * Mid-Up-Down pattern, with less maximas and minimas.
# * Having >3 Family Members slump the Chance of Survival.
# * The Chance of Survial is best with 1-3 Family Members.

# %% [markdown]
# #####  Survival / Embark Point

# %%
embark_survived_dist    = df_full[df_full.Survived==1].Embarked.value_counts().sort_index()
embark_deceased_dist    = df_full[df_full.Survived==0].Embarked.value_counts().sort_index()
embark_survival_dist    = pd.DataFrame([embark_survived_dist, embark_deceased_dist], index=SURVIVAL_LABEL).T
embark_survival_ratio   = embark_survival_dist.apply(lambda x: x/x.sum(), axis=1)

embark_survival_ratio.plot.bar()
plt.title("Survival Ratio / Embarkment Point ")
plt.show()

# %% [markdown]
# Vaguely Speaking, The Embarkment Point dosen't seem to be a big reason for variation in survival rate. But Visually it says a different story.
# <br>
# 
# People who embarked from C, have Higher Chance of survival Than those who Embarked from Other Places.

# %% [markdown]
# #####  Survival / Fare

# %%
df_full.Fare.describe().to_dict()

# %%
fare_bins = [0, 8, 14, 31, 513]
fare_Range_df = pd.cut(df_full.Fare, bins=fare_bins, include_lowest=True)

fare_grp_survived_dist    = fare_Range_df[df_full.Survived==1].value_counts().sort_index()
fare_grp_deceased_dist    = fare_Range_df[df_full.Survived==0].value_counts().sort_index()
fare_grp_survival_dist    = pd.DataFrame([fare_grp_survived_dist, fare_grp_deceased_dist], index=SURVIVAL_LABEL).T
fare_grp_survival_ratio   = fare_grp_survival_dist.apply(lambda x: x/x.sum(), axis=1)

fare_grp_survival_ratio.plot.bar()
plt.title("Survival Ratio / Class")
plt.show()

# %%
np.log(df_full.Fare.dropna() + 1e-1).astype(int)

# %%
fare_Range_df = np.log(df_full.Fare.dropna() + 1e-1).astype(int)

fare_log_grp_survived_dist    = fare_Range_df[df_full.Survived==1].value_counts().sort_index()
fare_log_grp_deceased_dist    = fare_Range_df[df_full.Survived==0].value_counts().sort_index()
fare_log_grp_survival_dist    = pd.DataFrame([fare_log_grp_survived_dist, fare_log_grp_deceased_dist], index=SURVIVAL_LABEL).T
fare_log_grp_survival_ratio   = fare_log_grp_survival_dist.apply(lambda x: x/x.sum(), axis=1)

fare_log_grp_survival_ratio.plot.bar()
plt.title("Survival Ratio / Fare Class")
plt.show()

# %% [markdown]
# Fare is related to Pclass, Higher the Fare, Higher The Class.<br>
# It is a Greate Feature to Use insted/along Pclass...<br><br>
# 
# It will be tested later.

# %% [markdown]
# ## Feature Engneering

# %% [markdown]
# ### Missing Dataset and Dealing With it.

# %%
sns.heatmap(data=df_full.notna(), cbar=False, cmap="Blues")

# %% [markdown]
# The data with "White" lines represent missing data and "Blue" lines represent not-null data.<br>
# We can either predict the missing data or leave the entire feature.

# %%
{k : i 
 for k, i in (m - df_full.notna().sum()).to_dict().items()
 if i}

# %% [markdown]
# Let's Now Predict the missing Values of **Embarked**, **Fare** and **Age**.

# %% [markdown]
# #### Embarked

# %%
df_full.Embarked.value_counts(normalize=1)

# %% [markdown]
# Since around 70% people Embarked from S.
# lets simply take the mode.

# %%
df_full.Embarked = df_full.Embarked.fillna(df_full.Embarked.mode()[0])

# %%
df_full.Embarked.isna().sum()

# %% [markdown]
# #### Fare

# %%
df_full[df_full.Fare.isna()]

# %% [markdown]
# We'll be choosing the mean of Fare in the group of people whose Pclass and Sex are the same.

# %%
grp_1 = df_full.groupby([PCLASS, SEX, IS_ALONE])
df_full.Fare = grp_1.Fare.apply(lambda fare: fare.fillna(fare.mean()))

# %% [markdown]
# #### Age

# %% [markdown]
# For Age Using only Sex and PClass may not be a good way, as this would easily skip all the children since the mean age is so high.<br>
# 
# Insted Lets Use Name. Yes, Name! to predict the age. <br>
# 
# We can use the title in the name as one of the feature to predict the age.

# %%
titles = df_full.Name.apply(lambda name: name.split(",")[1].split(".")[0].strip())
print(titles.unique())
titles[df_full.Age.isna()].value_counts()

# %% [markdown]
# We Can join and use some titles as one. <br>
# Though, This will increase the number of groups, at the same time this will also reduce edge case percentage.

# %%
male_titles     = titles[(df_full.Sex == "male")].unique()
female_titles   = titles[(df_full.Sex == "female")].unique()

unisex_titles       = [title_ for title_ in male_titles if title_ in female_titles]
male_only_titles    = [title_ for title_ in male_titles if title_ not in unisex_titles]
female_only_titles  = [title_ for title_ in female_titles if title_ not in unisex_titles]

unisex_titles, male_only_titles, female_only_titles

# %% [markdown]
# We can make 5 Categories - 
# * Mr,  Don,  Rev,  Major,  Sir,  Col,  Capt,  Jonkheer, [Dr]
# * Mrs,  Mme,  Lady,  Mlle,  the Countess,  Dona, [Dr]
# * Master
# * Miss, Ms (Those Who Traveled Alone, Implying They may be olded)
# * Miss, Ms (Those Who Traveled With Someone, Implying They may be younger)

# %%
Title_Cat_1 = ["Mr",  "Don",  "Rev",  "Major",  "Sir",  "Col",  "Capt",  "Jonkheer"]
Title_Cat_2 = ["Mrs",  "Mme",  "Lady",  "Mlle",  "the Countess",  "Dona"]
Title_Cat_3 = ["Master"]
Title_Cat_4 = ["Miss", "Ms"]


# %%
title_cat_num = titles.replace(Title_Cat_1, value="Adult Male")
title_cat_num = title_cat_num.replace(Title_Cat_2, "Adult Female")
title_cat_num = title_cat_num.replace(Title_Cat_3, "Young Boy")
title_cat_num = title_cat_num.replace(Title_Cat_4, "Young Female")

title_cat_num[(df_full.Sex=="male") & (title_cat_num=="Dr")]    = "Adult Male"
title_cat_num[(df_full.Sex=="female") & (title_cat_num=="Dr")]  = "Adult Female"
title_cat_num[(df_full.Is_Alone==0) & (title_cat_num=="Young Female")] = "Young Girl"

df_full[GENDER_AGE_CAT] = title_cat_num
df_full.Gender_Age_Cat.value_counts()

# %%
np.random.randn(200).max() / 3

# %%
grp_2 = df_full.groupby([GENDER_AGE_CAT, PCLASS])
df_full[AGE_PREDICT] = grp_2.Age.apply(lambda age: age.fillna(age.mean()))

overloaded_group = df_full[(df_full.Pclass==3) & (df_full.Gender_Age_Cat=="Adult Male")].Age
overloaded_group_index =  overloaded_group[df_full.Age.isna()].index


# %%
sns.heatmap(data=df_full.notna(), cbar=False, cmap="Blues")

# %%
df_full.isna().sum()

# %% [markdown]
# Now, We have Predicted the missing values, Weather we should use age, is a difficult to answer question...
# We'll compare the results in later stages.

# %% [markdown]
# ### Feature Encoding

# %% [markdown]
# In This Part, We'll Encode the Features... <br>
# 
# * One Hot: Multi Categorical Features
# * Label Encoding: Binary Categorical Features

# %%
df_full.columns

# %%
df_full[[*FEATURES_1, *FEATURES_2]].nunique()

# %% [markdown]
# <!-- * Binary Categorical Features - Survived, Sex, Is_Alone
# * Multi Categorical Features - Pclass, Embarked, Gender_Age_Cat
# * Continous Features - Age, SibSp, Parch, Fare, Age_Predict -->
# 
# |Binary Categorical Features|Multi Categorical Features|Continous Features|
# |---|---|---|
# |  ***Survived***, ***Sex***, ***Is_Alone***  |  ***Pclass***, ***Embarked***, ***Gender_Age_Cat***  |  ***Age***, ***SibSp***, ***Parch***, ***Fare***, ***Age_Predict***  |

# %%
def data_encoder(df_: pd.DataFrame):    
    df = df_.copy()
    df.Sex = (df.Sex=="male").astype(int)
    
    df["Pclass_1"] = (df.Pclass==1).astype(int)
    df["Pclass_2"] = (df.Pclass==2).astype(int)
    
    df["Embarked_From_S"] = (df.Embarked=="S").astype(int)
    df["Embarked_From_C"] = (df.Embarked=="C").astype(int)
    
    return df.drop([EMBARKED, PCLASS], axis=1)

df_full_encoded = data_encoder(df_full.drop([*FEATURES_USELESS, AGE, NAME, GENDER_AGE_CAT], axis=1))
features = df_full_encoded.columns[1:]
df_full_encoded.head(10)

# %% [markdown]
# We have Sucessfully Encoded the features and removed the redundent features.

# %% [markdown]
# ## Train Dev Test devide

# %%
df_train, df_test = train_test_devide(df_full_encoded)

# %%
from sklearn.model_selection import train_test_split

def train_test_devide(df_: pd.DataFrame) -> tuple[pd.DataFrame]:
    df_train_ = df_[(df_full.Survived.notna())]
    df_test_ = df_[(df_full.Survived.isna())]
    return df_train_, df_test_

def xy_devide(df_: pd.DataFrame) -> tuple[np.ndarray]:
    X = df_.iloc[:, 1:].values
    y = df_.iloc[:, 0].values
    return X, y

def train_dev_test_devide(df_: pd.DataFrame, train_size_: float=0.8, random_state_: int=0) -> tuple[np.ndarray]:
    df_train_, df_test_ = train_test_devide(df_)
    X, y, = xy_devide(df_train_)
    X_train_, X_Dev_, y_train_, y_Dev_ = train_test_split(X, y, 
                                                      train_size=train_size_, 
                                                      random_state=random_state_)
    X_test_, y_test_ = xy_devide(df_test_)
    
    return X_train_, X_Dev_, X_test_, y_train_, y_Dev_, y_test_


X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_devide(df_full_encoded)

# %%
print(list(map(lambda x: x.shape, [df_full, df_train, df_test])))
list(map(lambda x: x.shape, [X_train, X_dev, X_test, y_train, y_dev, y_test]))

# %% [markdown]
# ## Feature Scaling

# %% [markdown]
# In This Part, We'll Scale the Features.
# 
# * Min-Max Scaling.
# * Absolute Max Scaling.
# * Normalize.
# * Standardization.

# %%
df_full_encoded.columns[1:]

# %%
X_train[:5]

# %% [markdown]
# We have encoded the features and converted them into a numpy object. <br>
# 
# Let's Scale it.

# %%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
indexes = [1, 2, 3, 5]

X_train_scaled  = X_train.copy()
X_test_scaled   = X_test.copy()
X_dev_scaled    = X_dev.copy()

X_train_scaled[:, indexes]  = sc.fit_transform(X_train[:, indexes])
X_test_scaled[:, indexes]   = sc.transform(X_test[:, indexes])
X_dev_scaled[:, indexes]    = sc.transform(X_dev[:, indexes])

# %%
print(X_train_scaled[:1], "\n")
print(X_dev_scaled[:1], "\n")
print(X_test_scaled[:1], "\n")

# %% [markdown]
# Now, we have Scaled the data, Let's beguin the Basic Predictions

# %% [markdown]
# # Predictions
# 
# In this Section, We'll Use Various Models their Hyper Parameters to see which one's better.

# %% [markdown]
# ### Models: <br>
# 
# 1. Logistic Regression
# 2. Knn
# 3. SVM
# 4. Desecion Tree
# 5. Random Forest
# 6. Gaussian Naive Bayes
# 7. ANN

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

def get_model_score_stats(y_: np.ndarray, y_pred_: np.ndarray) -> tuple[float]:
    ac  = accuracy_score(y_, y_pred_)
    rcl = recall_score(y_, y_pred_)
    pcn = precision_score(y_, y_pred_)
    f1  = f1_score(y_, y_pred_)
    
    return ac, rcl, pcn, f1

def get_confussion_matrix(y_: np.ndarray, y_pred_: np.ndarray):
    cm = confusion_matrix(y_.tolist(), y_pred_.tolist())
    sns.heatmap(cm, annot=True, cmap="mako")
    
def join_array(y_: np.ndarray, y_pred_: np.ndarray) -> np.ndarray:
    m = y_.shape[0]
    return np.concatenate(
        [y_.reshape(m, 1), y_pred_.reshape(m, 1)], 
        axis=1
    )

# %% [markdown]
# #### Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression

log_reg_clf = LogisticRegression()
log_reg_clf.fit(X_train_scaled, y_train)

y_pred_log = log_reg_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_log)

plt.title("Comfusion Matrix for Logistic Regression", fontsize=15)
plt.show()

# %%
accuracy_log_reg, recal_log_reg, precision_log_reg, fi_log_reg = get_model_score_stats(y_dev, y_pred_log)
accuracy_log_reg, recal_log_reg, precision_log_reg, fi_log_reg

# %% [markdown]
# #### Knn

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_scaled, y_train)

y_pred_knn = knn_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_knn)

plt.title("Comfusion Matrix for K Nearest Neighbors", fontsize=15)
plt.show()

# %%
accuracy_knn, recal_knn, precision_knn, fi_knn = get_model_score_stats(y_dev, y_pred_knn)
accuracy_knn, recal_knn, precision_knn, fi_knn

# %% [markdown]
# #### SVM

# %%
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)

y_pred_svm = svm_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_svm)

plt.title("Comfusion Matrix for Scaler Vector Classifier", fontsize=15)
plt.show()

# %%
accuracy_svm, recal_svm, precision_svm, fi_svm = get_model_score_stats(y_dev, y_pred_svm)
accuracy_svm, recal_svm, precision_svm, fi_svm

# %% [markdown]
# #### Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

dcn_tree_clf = DecisionTreeClassifier(random_state=0)
dcn_tree_clf.fit(X_train_scaled, y_train)

y_pred_dcn_tree = dcn_tree_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_dcn_tree)

plt.title("Comfusion Matrix for Decision Tree", fontsize=15)
plt.show()

# %%
accuracy_dcn_tree, recal_dcn_tree, precision_dcn_tree, fi_dcn_tree = get_model_score_stats(y_dev, y_pred_dcn_tree)
accuracy_dcn_tree, recal_dcn_tree, precision_dcn_tree, fi_dcn_tree

# %% [markdown]
# #### Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

rdm_fst_clf = RandomForestClassifier(random_state=0)
rdm_fst_clf.fit(X_train_scaled, y_train)

y_pred_rdm_fst = rdm_fst_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_rdm_fst)

plt.title("Comfusion Matrix for Random Forest", fontsize=15)
plt.show()

# %%
accuracy_rdm_fst, recal_rdm_fst, precision_rdm_fst, fi_rdm_fst = get_model_score_stats(y_dev, y_pred_rdm_fst)
accuracy_rdm_fst, recal_rdm_fst, precision_rdm_fst, fi_rdm_fst

# %% [markdown]
# #### Naive Bayes

# %%
from sklearn.naive_bayes import GaussianNB

gsn_nb_clf = GaussianNB()
gsn_nb_clf.fit(X_train_scaled, y_train)

y_pred_gsn_nb = gsn_nb_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_gsn_nb)

plt.title("Comfusion Matrix for Logistic Regression", fontsize=15)
plt.show()

# %%
accuracy_nb, recal_nb, precision_nb, fi_nb = get_model_score_stats(y_dev, y_pred_gsn_nb)

# %% [markdown]
# #### ANN

# %%
from sklearn.neural_network import MLPClassifier

ann_clf = MLPClassifier(solver='adam')
ann_clf.fit(X_train_scaled, y_train)

y_pred_ann = ann_clf.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_ann)

plt.title("Comfusion Matrix for Logistic Regression", fontsize=15)
plt.show()

# %%
get_model_score_stats(y_dev, y_pred_ann)

# %% [markdown]
# #### Model Comparision

# %%
model_comp_df = pd.DataFrame({
    "Model":            ["Log Reg", "KNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes"],
    "Accuracy Score":   [accuracy_log_reg, accuracy_knn, accuracy_svm, accuracy_dcn_tree, accuracy_rdm_fst, accuracy_nb],
    "Recal":            [recal_log_reg, recal_knn, recal_svm, recal_dcn_tree, recal_rdm_fst, recal_nb],
    "Precision":        [precision_log_reg, precision_knn, precision_svm, precision_dcn_tree, precision_rdm_fst, precision_nb],
    "F1 Score":         [fi_log_reg, fi_knn, fi_svm, fi_dcn_tree, fi_rdm_fst, fi_nb],
    
})

model_comp_df = model_comp_df.sort_values(by="Accuracy Score", ascending=False)
model_comp_df = model_comp_df.set_index("Model")
model_comp_df

# %%
sns.heatmap(model_comp_df, annot=True, cmap="mako")
plt.title("Model Comparison Table", fontsize=15)

# %% [markdown]
# ### Hyperparameter Tuning

# %%
from sklearn.model_selection import KFold, GridSearchCV

# %%
X_scaled    = np.concatenate([X_train_scaled, X_dev_scaled])
y           = np.concatenate([y_train, y_dev])

# %% [markdown]
# #### Logistic Regression

# %%
hyperpars_lr = {
    "max_iter":     [20, 50, 100, 150, 200, 500],
    "penalty":      ["l1", "l2"],
    "C":            [100, 10, 1.0, 0.1, 0.01],
    "class_weight": ["balanced", None],
    "solver":       ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],   
}

folds = KFold(n_splits=5, shuffle=True, random_state=1)
log_reg_clf_2 = LogisticRegression()

log_reg_grid_search = GridSearchCV(estimator=log_reg_clf_2, 
                                   param_grid=hyperpars_lr, 
                                   verbose=1, 
                                   cv=folds, 
                                   n_jobs=-1)

log_reg_grid_search.fit(X_scaled, y)


# %%
best_score_log_reg          = log_reg_grid_search.best_score_
best_hyperparams_log_reg    = log_reg_grid_search.best_params_

best_hyperparams_log_reg, best_score_log_reg


# %%
log_reg_clf_2 = LogisticRegression(**best_hyperparams_log_reg)
log_reg_clf_2.fit(X_train_scaled, y_train)

y_pred_log_reg_2 = log_reg_clf_2.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_log_reg_2)

plt.title("Comfusion Matrix for Tuned Logistic Regression", fontsize=15)
plt.show()

# %%
accuracy_log_reg_2, recal_log_reg_2, precision_log_reg_2, fi_log_reg_2 = get_model_score_stats(y_dev, y_pred_log_reg_2)
accuracy_log_reg_2, recal_log_reg_2, precision_log_reg_2, fi_log_reg_2

# %% [markdown]
# #### KNN

# %%
hyperpars_knn = {
    "n_neighbors":  list(range(1, 8)),
    "leaf_size":    list(range(20, 40)),
    "p":            [1, 2],
}

folds = KFold(n_splits=5, shuffle=True, random_state=1)

knn_grid_search = GridSearchCV(estimator=KNeighborsClassifier(), 
                                   param_grid=hyperpars_knn, 
                                   verbose=1, 
                                   cv=folds, 
                                   n_jobs=-1,
                                   )

knn_grid_search.fit(X_scaled, y)


# %%
best_score_knn = knn_grid_search.best_score_
best_hyperparams_knn = knn_grid_search.best_params_

best_hyperparams_knn, best_score_knn


# %%
knn_clf_2 = KNeighborsClassifier(**best_hyperparams_knn)
knn_clf_2.fit(X_train_scaled, y_train)

y_pred_knn_2 = knn_clf_2.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_knn_2)

plt.title("Comfusion Matrix for Tuned K Nearest Neighbors", fontsize=15)
plt.show()

# %%
accuracy_knn_2, recal_knn_2, precision_knn_2, fi_knn_2 = get_model_score_stats(y_dev, y_pred_knn_2)
accuracy_knn_2, recal_knn_2, precision_knn_2, fi_knn_2

# %% [markdown]
# #### SVM

# %%
hyperpars_svm = {
    "C":        [1000, 100, 10, 1.0, 0.1],
    "gamma":    [1e-1, 1e-2, 1e-3, 1e-4],
    "kernel":   ["linear", "poly", "rbf", "sigmoid"],
}

folds = KFold(n_splits=5, shuffle=True, random_state=1)

svm_grid_search = GridSearchCV(estimator=SVC(random_state=1), 
                                   param_grid=hyperpars_svm, 
                                   verbose=1, 
                                   cv=folds, 
                                   n_jobs=-1)

svm_grid_search.fit(X_scaled, y)


# %%
best_score_svm = svm_grid_search.best_score_
best_hyperparams_svm = svm_grid_search.best_params_

best_hyperparams_svm, best_score_svm


# %%
svm_clf_2 = SVC(**best_hyperparams_svm)
svm_clf_2.fit(X_train_scaled, y_train)

y_pred_svm_2 = svm_clf_2.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_svm_2)

plt.title("Comfusion Matrix for Tuned Scaler Vector Classifier", fontsize=15)
plt.show()

# %%
accuracy_svm_2, recal_svm_2, precision_svm_2, fi_svm_2 = get_model_score_stats(y_dev, y_pred_svm_2)
accuracy_svm_2, recal_svm_2, precision_svm_2, fi_svm_2

# %% [markdown]
# #### Decision Tree

# %%
hyperpars_dcn_tree = {
    "criterion":    ["gini", "entropy"],
    "splitter":     ["best", "random"],
    "max_depth":    [*list(range(1, 10)), None],
    # "min_samples_leafint":    [*list(range(1, 5))]
}

folds = KFold(n_splits=5, shuffle=True, random_state=1)

dcn_tree_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1), 
                                   param_grid=hyperpars_dcn_tree, 
                                   verbose=1, 
                                   cv=folds, 
                                   n_jobs=-1)

dcn_tree_grid_search.fit(X_scaled, y)


# %%
best_score_dcn_tree = dcn_tree_grid_search.best_score_
best_hyperparams_dcn_tree = dcn_tree_grid_search.best_params_

best_hyperparams_dcn_tree, best_score_dcn_tree


# %%
dcn_tree_clf_2 = DecisionTreeClassifier(**best_hyperparams_dcn_tree)
dcn_tree_clf_2.fit(X_train_scaled, y_train)

y_pred_dcn_tree_2 = dcn_tree_clf_2.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_dcn_tree_2)

plt.title("Comfusion Matrix for Tuned Decision Tree", fontsize=15)
plt.show()

# %%
accuracy_dcn_tree_2, recal_dcn_tree_2, precision_dcn_tree_2, fi_dcn_tree_2 = get_model_score_stats(y_dev, y_pred_dcn_tree_2)
accuracy_dcn_tree_2, recal_dcn_tree_2, precision_dcn_tree_2, fi_dcn_tree_2

# %% [markdown]
# #### Random Forest

# %%
hyperpars_rdm_fst = {
    "n_estimators":     list(range(85, 100)),
    "max_depth":        range(6, 10),
    # "criterion":        ["gini", "entropy"],
}

folds = KFold(n_splits=5, shuffle=True, random_state=1)

rdm_fst_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=1), 
                                   param_grid=hyperpars_rdm_fst, 
                                   verbose=1, 
                                   cv=folds, 
                                   n_jobs=-1)

rdm_fst_grid_search.fit(X_scaled, y)

# %%
best_score_rdm_fst = rdm_fst_grid_search.best_score_
best_hyperparams_rdm_fst = rdm_fst_grid_search.best_params_

best_hyperparams_rdm_fst, best_score_rdm_fst


# %%
rdm_fst_clf_2 = RandomForestClassifier(**best_hyperparams_rdm_fst ,random_state=0)
rdm_fst_clf_2.fit(X_train_scaled, y_train)

y_pred_rdm_fst_2 = rdm_fst_clf_2.predict(X_dev_scaled)

# %%
get_confussion_matrix(y_dev, y_pred_rdm_fst_2)

plt.title("Comfusion Matrix for Tuned Random Forest", fontsize=15)
plt.show()

# %%
accuracy_rdm_fst_2, recal_rdm_fst_2, precision_rdm_fst_2, fi_rdm_fst_2 = get_model_score_stats(y_dev, y_pred_rdm_fst_2)
accuracy_rdm_fst_2, recal_rdm_fst_2, precision_rdm_fst_2, fi_rdm_fst_2

# %% [markdown]
# #### Model Comparison

# %%
model_comp_df_2 = pd.DataFrame({
    "Model":            ["Log Reg Tuned", "KNN Tuned", "SVM Tuned", "Decision Tree Tuned", "Random Forest Tuned"],
    "Accuracy Score":   [accuracy_log_reg_2, accuracy_knn_2, accuracy_svm_2, accuracy_dcn_tree_2, accuracy_rdm_fst_2],
    "Recal":            [recal_log_reg_2, recal_knn_2, recal_svm_2, recal_dcn_tree_2, recal_rdm_fst_2],
    "Precision":        [precision_log_reg_2, precision_knn_2, precision_svm_2, precision_dcn_tree_2, precision_rdm_fst_2],
    "F1 Score":         [fi_log_reg_2, fi_knn_2, fi_svm_2, fi_dcn_tree_2, fi_rdm_fst_2],
    
})

model_comp_df_2 = model_comp_df_2.sort_values(by="Accuracy Score", ascending=False)
model_comp_df_2 = model_comp_df_2.set_index("Model")
model_comp_df_2

# %%
model_comp_df_all = pd.concat([model_comp_df_2, model_comp_df])
model_comp_df_all = model_comp_df_all.sort_values(by="Accuracy Score", ascending=False)

model_comp_df_all

# %%
sns.heatmap(model_comp_df_all, annot=True, cmap="mako")
plt.title("All Models Comparison Table", fontsize=15)

# %%
best_hyperparams_rdm_fst

# %%
df_test

# %%
final_model = SVC(**best_hyperparams_svm)
final_model.fit(X_scaled, y)
predictions = final_model.predict(X_test_scaled).astype(int)

# output = pd.DataFrame({'PassengerId': (df_test.index+1), 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")

# %%
actual_result= pd.read_csv(submission_data_path).values[:, 1]

actual_result.shape

# %%
get_confussion_matrix(actual_result, predictions)

# %%
get_model_score_stats(actual_result, predictions)

# %%


# %%
time() - init_time

