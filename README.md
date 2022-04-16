<div class="cell markdown" id="0FPrU6a7FWTx">

Importing Packages

</div>

<div class="cell code" data-execution_count="144" id="AbIWk6Ns2AGO">

``` python
#Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
```

</div>

<div class="cell markdown" id="jqAvvKC6FbP-">

Connecting collab to drive

</div>

<div class="cell code" data-execution_count="145" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="AExa8bCh5agk" data-outputId="efc9553c-d1f3-4f6b-9ddd-a5cd3238b441">

``` python
from google.colab import drive
drive.mount('/content/drive')
```

<div class="output stream stdout">

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

</div>

</div>

<div class="cell markdown" id="fAfF1NyYFk7h">

Importing data from drive

</div>

<div class="cell code" data-execution_count="146" id="JUAgjYS75dB6">

``` python
df = pd.read_csv('/content/drive/My Drive/Stresslevel.csv')
```

</div>

<div class="cell markdown" id="dHlMx_F5FxBx">

Defining the data on Inputs and Outputs

</div>

<div class="cell code" data-execution_count="147" id="C6DD6tD7TaNx">

``` python
X = df.drop('Stress Level', axis=1)
y = df['Stress Level']
```

</div>

<div class="cell markdown" id="Czjpsoq3F6I1">

Taking a look at the data

</div>

<div class="cell code" data-execution_count="148" data-colab="{&quot;height&quot;:206,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ZbFOFEzs5uFn" data-outputId="069faae1-d415-4218-be8c-82e3dca497f3">

``` python
df.head()
```

<div class="output execute_result" data-execution_count="148">

``` 
   Humidity  Temperature  Step count  Stress Level
0     21.33        90.33         123             1
1     21.41        90.41          93             1
2     27.12        96.12         196             2
3     27.64        96.64         177             2
4     10.87        79.87          87             0
```

</div>

</div>

<div class="cell markdown" id="niZWAujXF93F">

Shape of the data

</div>

<div class="cell code" data-execution_count="149" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="wYcrMvBH63xt" data-outputId="55e99142-5709-4848-e59a-c6c55b495d19">

``` python
df.shape
```

<div class="output execute_result" data-execution_count="149">

    (2001, 4)

</div>

</div>

<div class="cell markdown" id="1UDvUauUGsTn">

headers of the data

</div>

<div class="cell code" data-execution_count="155" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="HeRXU7XlLbRU" data-outputId="033ff593-4d7c-4c4e-8bd9-df61db0a1782">

``` python
print(df.columns.values[0:4])
```

<div class="output stream stdout">

    ['Humidity' 'Temperature' 'Step count' 'Stress Level']

</div>

</div>

<div class="cell markdown" id="8JpWM2YaGxxm">

Checking the null values

</div>

<div class="cell code" data-execution_count="156" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="bAy2QouF6_UH" data-outputId="97bd9801-edc8-42d6-9ed4-a0270ee2790e">

``` python
print(df.isnull().sum())
```

<div class="output stream stdout">

    Humidity        0
    Temperature     0
    Step count      0
    Stress Level    0
    dtype: int64

</div>

</div>

<div class="cell markdown" id="4KDvQdDPG-Ih">

# Describing each stress level

</div>

<div class="cell code" data-execution_count="157" data-colab="{&quot;height&quot;:175,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="0Z_qVU4nL2zn" data-outputId="918b5cdb-ac86-4d43-ad7e-ccb984617897">

``` python
# Descriptive statistics of the least stressed level (0)

summary = (df[df['Stress Level'] == 0].describe().transpose().reset_index())
summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)
val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
summary
```

<div class="output execute_result" data-execution_count="157">

``` 
        feature  count    mean     std   min    25%   50%    75%   max
0      Humidity  501.0  12.500   1.448  10.0  11.25  12.5  13.75  15.0
1   Temperature  501.0  81.500   1.448  79.0  80.25  81.5  82.75  84.0
2    Step count  501.0  42.934  26.199   0.0  20.00  41.0  65.00  90.0
3  Stress Level  501.0   0.000   0.000   0.0   0.00   0.0   0.00   0.0
```

</div>

</div>

<div class="cell code" data-execution_count="158" data-colab="{&quot;height&quot;:175,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="eMibfqxVMsmG" data-outputId="34829a3a-9146-4a40-bd31-8fac0135cc92">

``` python
# Descriptive statistics of the middle stressed level (1)

summary = (df[df['Stress Level'] == 1].describe().transpose().reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
summary
```

<div class="output execute_result" data-execution_count="158">

``` 
        feature  count    mean     std    min     25%     50%      75%    max
0      Humidity  790.0  18.955   2.282  15.01  16.982  18.955   20.928   22.9
1   Temperature  790.0  87.955   2.282  84.01  85.982  87.955   89.928   91.9
2    Step count  790.0  78.130  37.677   0.00  48.000  88.000  110.000  129.0
3  Stress Level  790.0   1.000   0.000   1.00   1.000   1.000    1.000    1.0
```

</div>

</div>

<div class="cell code" data-execution_count="159" data-colab="{&quot;height&quot;:175,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="jFQt2UAvMgmu" data-outputId="d448c536-cc4a-45df-93f3-6ecffffe1ec7">

``` python
# Descriptive statistics of the most stressed level (2)

summary = (df[df['Stress Level'] == 2].describe().transpose().reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
summary
```

<div class="output execute_result" data-execution_count="159">

``` 
        feature  count     mean     std     min      25%      50%      75%  \
0      Humidity  710.0   26.455   2.051   22.91   24.682   26.455   28.228   
1   Temperature  710.0   95.455   2.051   91.91   93.682   95.455   97.228   
2    Step count  710.0  165.000  20.508  130.00  147.000  165.000  183.000   
3  Stress Level  710.0    2.000   0.000    2.00    2.000    2.000    2.000   

     max  
0   30.0  
1   99.0  
2  200.0  
3    2.0  
```

</div>

</div>

<div class="cell markdown" id="5cI50rJqHRFl">

checking if there is any imbalance data

</div>

<div class="cell code" data-colab="{&quot;height&quot;:388,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="2AqKNdOc_6JP" data-outputId="faea3adf-b3a8-451a-d1fa-78fde27c47a3">

``` python
sns.countplot(x="Stress Level", data =df)
plt.show()
```

<div class="output display_data">

![](13f83529e2256fad1dec75f3d95eccb765a1c296.png)

</div>

</div>

<div class="cell code" id="G2k9z-gyHnK3">

``` python
# We can see that data is not imbalance
```

</div>

<div class="cell markdown" id="wCSDQs4QHvCu">

Plot a pie chart for different stress levels

</div>

<div class="cell code" data-colab="{&quot;height&quot;:410,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="SXLh47AEBIQh" data-outputId="d2eae4fe-908f-46b1-c787-c48ca8329207">

``` python
stress_level = df["Stress Level"].value_counts().tolist()
labels = ['0','1', '2']
values = [stress_level[0], stress_level[1], stress_level[2]]
print(values)
colors = ['lightgreen', 'lightblue', 'purple']
fix, ax1 = plt.subplots()
# the autopct is to get the percentage values.
_, texts, autotexts = ax1.pie(values, labels=labels, colors=colors, startangle = 90,shadow=True, autopct='%1.1f%%')
list(map(lambda x:x.set_fontsize(15), autotexts))
ax1.set_title("Stress Levels", fontsize=15)
```

<div class="output stream stdout">

    [790, 710, 501]

</div>

<div class="output execute_result" data-execution_count="126">

    Text(0.5, 1.0, 'Stress Levels')

</div>

<div class="output display_data">

![](aeb777d4248e6e907149987858148fd6c78c416f.png)

</div>

</div>

<div class="cell markdown" id="hTd4bWJlH8BU">

Correlation of variables with the Stress Level

</div>

<div class="cell code" data-execution_count="160" data-colab="{&quot;height&quot;:884,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="RQyIqGrdKuYM" data-outputId="14acd3d5-975b-4fb8-fd32-15058636fe7f">

``` python
corr = df.corrwith(df['Stress Level']).reset_index()

corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = False)

plt.figure(figsize=(8, 15))
fig = sns.heatmap(corr, annot=True, fmt="g", linewidths=0.4)

plt.title("Correlation of Variables with Stress Level", fontsize=20)
plt.show()
```

<div class="output display_data">

![](cc20c6db222657c13f2f258a90a6afd2727b8c2f.png)

</div>

</div>

<div class="cell code" data-execution_count="161" data-colab="{&quot;height&quot;:630,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="084DLzCKNa8Y" data-outputId="daa37cec-b80d-4b7c-9ac9-66efcb68fb20">

``` python
#Correlation of variables with eachother
plt.figure(figsize=(15,10))
plt.title("Correlation of Variables with each other", fontsize=20 )
sns.heatmap(df.corr(), annot = True, fmt = '.1f' )
```

<div class="output execute_result" data-execution_count="161">

    <matplotlib.axes._subplots.AxesSubplot at 0x7f8f6b45d850>

</div>

<div class="output display_data">

![](bfdee7c77cb64da2ff627f36e49e807a315cdefb.png)

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:294,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="V6-o_KK-N2Nk" data-outputId="f379f1e4-26df-470d-dea0-b603fb70e8ae">

``` python
#Creating boxplot
f, axes = plt.subplots(ncols=3, figsize=(20,4))

sns.boxplot(x="Stress Level", y="Humidity", data=df, palette=colors, ax=axes[0])
axes[0].set_title('Humidity vs Stress Level (Positive Correlation)')

sns.boxplot(x="Stress Level", y="Temperature", data=df, palette=colors, ax=axes[1])
axes[1].set_title('Temperature vs Stress Level (Positive Correlation)')

sns.boxplot(x="Stress Level", y="Step count", data=df, palette=colors, ax=axes[2])
axes[2].set_title('Step Count vs Stress Level (Positive Correlation)')

plt.show()
```

<div class="output display_data">

![](66f8a0f3e66a4b77634c3340971fe6766ba6562b.png)

</div>

</div>

<div class="cell markdown" id="4L7V0UbVITSo">

Plotting T-SNE

</div>

<div class="cell code" data-colab="{&quot;height&quot;:860,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="kCobUd2ySEp7" data-outputId="9cc300c8-109b-4478-973c-84b0db2c8fd3">

``` python
from sklearn.manifold import TSNE

X_for_tsne = df.drop(['Stress Level'], axis=1)

# Commented out IPython magic to ensure Python compatibility.
# %time
tsne = TSNE(random_state = 42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(X_for_tsne)

plt.figure(figsize=(14,10))
sns.scatterplot(x =tsne[:, 0], y = tsne[:, 1], hue = df["Stress Level"],palette="bright")
```

<div class="output stream stderr">

    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:783: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.
      FutureWarning,
    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:793: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      FutureWarning,

</div>

<div class="output stream stdout">

    [t-SNE] Computing 151 nearest neighbors...
    [t-SNE] Indexed 2001 samples in 0.002s...
    [t-SNE] Computed neighbors for 2001 samples in 0.046s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 2001
    [t-SNE] Computed conditional probabilities for sample 2000 / 2001
    [t-SNE] Computed conditional probabilities for sample 2001 / 2001
    [t-SNE] Mean sigma: 2.757360
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 51.331356
    [t-SNE] KL divergence after 1000 iterations: 0.359171

</div>

<div class="output execute_result" data-execution_count="130">

    <matplotlib.axes._subplots.AxesSubplot at 0x7f8f68a51fd0>

</div>

<div class="output display_data">

![](f60c3e41ec7bdfa36f6e2ec6b3e436ddf2048f7b.png)

</div>

</div>

<div class="cell markdown" id="5IKK-Nz7Igfi">

Updating Inputs and Outputs with Standard scalar to check if there is
any change

</div>

<div class="cell code" data-execution_count="163" id="o9iz5Tm5x7TI">

``` python
standard_scaler = StandardScaler()
X = df.drop('Stress Level', axis=1)
y = df['Stress Level']
#Scale features to improve the training ability of TSNE.
X_Scaled = standard_scaler.fit_transform(X.values)
```

</div>

<div class="cell code" data-execution_count="164" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="aDXAXdJ5b7nC" data-outputId="c21eb42a-a6c7-49b0-8d26-257035fa74d7">

``` python
# t-SNE

X_reduced_tsne = TSNE(n_components=2, random_state=2).fit_transform(X_Scaled)
```

<div class="output stream stderr">

    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:783: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.
      FutureWarning,
    /usr/local/lib/python3.7/dist-packages/sklearn/manifold/_t_sne.py:793: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      FutureWarning,

</div>

</div>

<div class="cell markdown" id="kqHJsqU7Isgx">

t-SNE after StandardScaler

</div>

<div class="cell code" data-execution_count="165" data-colab="{&quot;height&quot;:404,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="iIcQnPQbcKZl" data-outputId="a5e66446-b430-4ae7-f417-5a9672f8ad65">

``` python
color_map = {2:'purple', 1:'lightblue', 0:'lightgreen'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X_reduced_tsne[y==cl,0], 
                y=X_reduced_tsne[y==cl,1], 
                c=color_map[idx], 
                label=cl)
plt.xlabel('X ')
plt.ylabel('Y ')
plt.legend(loc='best')
plt.title('t-SNE visualization')
plt.show()
```

<div class="output display_data">

![](2db2653a7e0df7f853f2ee3c7cc26e06191f46a2.png)

</div>

</div>

<div class="cell markdown" id="THCfrDV4xMO-">

Implementing Algo

</div>

<div class="cell markdown" id="j7-IrFsvI09l">

Support vector Machine

</div>

<div class="cell code" id="HIgCf30xxWg2">

``` python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # SVM algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # evaluation metric
import itertools # advanced tools
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```

</div>

<div class="cell code" id="Mom9FOQKxQaX">

``` python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=0)
```

</div>

<div class="cell code" id="3y51Ju01yTvQ">

``` python
#Without standard scalar

svm = SVC()
svm.fit(X_train, y_train)
svm_ypred = svm.predict(X_test)
```

</div>

<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="-op4grZoy1gd" data-outputId="80f6b3b1-9a63-4b60-d58a-bb29f2144437">

``` python
print('Accuracy score of the SVM model is :'+str(accuracy_score(y_test, svm_ypred)))
```

<div class="output stream stdout">

    Accuracy score of the SVM model is :1.0

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:441,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="ucuU77cC7Quz" data-outputId="9148f033-0bde-4157-dca6-9e035c2bc1e9">

``` python
def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

svm_matrix = confusion_matrix(y_test, svm_ypred) # Support Vector Machine

plt.rcParams['figure.figsize'] = (6, 6)

svm_cm_plot = plot_confusion_matrix(svm_matrix, 
                                classes = ['0','1','2'], 
                                normalize = False, title = 'SVM')
plt.savefig('svm_cm_plot.png')
plt.show()
```

<div class="output display_data">

![](4f4aa7b372273fe7c6654a42ef790f7241a5afa8.png)

</div>

</div>

<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="MdLSSa3303jd" data-outputId="de7de5e8-5eb0-465b-a29e-9384fe4bc6c0">

``` python
#Predict the stress with user input values
print(svm.predict([[20,80,200]]))
```

<div class="output stream stdout">

    [2]

</div>

<div class="output stream stderr">

    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but SVC was fitted with feature names
      "X does not have valid feature names, but"

</div>

</div>

<div class="cell code" id="NcoFmprb3uDO">

``` python
#With standard scalar

X_train= standard_scaler.fit_transform(X_train)
X_test=standard_scaler.transform(X_test)

svm = SVC()
svm.fit(X_train, y_train)
svm_ypred = svm.predict(X_test)
```

</div>

<div class="cell code" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="b2xbi1cW3-Pl" data-outputId="99484a6e-1f58-480b-ec00-038a42f663cb">

``` python
print('Accuracy score of the SVM model is :'+str(accuracy_score(y_test, svm_ypred)))
```

<div class="output stream stdout">

    Accuracy score of the SVM model is :0.9975062344139651

</div>

</div>

<div class="cell code" data-colab="{&quot;height&quot;:441,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="wTsisG_X8_nx" data-outputId="a1ed75dc-0310-4da0-f213-4f0bcb9ec6c8">

``` python
def plot_confusion_matrix(cm, classes, title, normalize = False, cmap = plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

svm_matrix = confusion_matrix(y_test, svm_ypred) # Support Vector Machine

plt.rcParams['figure.figsize'] = (6, 6)

svm_cm_plot = plot_confusion_matrix(svm_matrix, 
                                classes = ['0','1','2'], 
                                normalize = False, title = 'SVM')
plt.savefig('svm_cm_plot.png')
plt.show()
```

<div class="output display_data">

![](d117f9511cca6e19bde2d337e71a3caaa4fdd884.png)

</div>

</div>
