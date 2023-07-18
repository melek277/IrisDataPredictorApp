#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 


# In[2]:


import pandas as pd 


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


import sklearn.datasets as datasets


# In[6]:


from sklearn import metrics 


# In[7]:


iris = datasets.load_iris()


# In[8]:


data=pd.DataFrame({
'sepal length': iris.data[:,0],
'sepal width': iris.data[:,1],
'petal length': iris.data[:,2],
'petal width': iris.data[:,3], 
'species': iris.target})
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']] #features
y=data['species']  #target


# In[9]:


data["species"].unique()


# In[10]:


x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.3) #splitting data with test size of 30%


# In[11]:


clf=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 10 decision trees


# In[12]:


clf.fit(x_train, y_train)  #Training our model


# In[13]:


y_pred=clf.predict(x_test)  #testing our model


# In[14]:




# In[15]:


print(data["sepal length"].max())
print(data["sepal length"].min())


# In[16]:


print(data["sepal width"].max())
print(data["sepal width"].min())


# In[17]:


print(data["petal length"].max())
print(data["petal length"].min())


# In[18]:


print(data["petal width"].max())
print(data["petal width"].min())


# In[19]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  #Measuring the accuracy of our model


# In[20]:


st.title('Iris Data Predictor')


# In[21]:





# In[22]:


SepalLength=st.slider("Fill the Sepal Length",min_value=0.0, max_value=10.0, step=0.01, value=5.84)
SepalWidth=st.slider("Fill the Sepal Width",min_value=0.0, max_value=5.0, step=0.01, value=3.05)
PetalLength=st.slider("Fill the Petal Length",min_value=0.0, max_value=10.0, step=0.01, value=3.75)
PetalWidth=st.slider("Fill the Petal Width",min_value=0.0, max_value=5.0, step=0.01, value=1.19)


# In[23]:


button=st.button("Predict",use_container_width=True  )


# In[24]:


index = [1, 2, 3, 4]
df = pd.DataFrame({
'sepal length': SepalLength,
'sepal width': SepalWidth,
'petal length': PetalLength,
'petal width':  PetalWidth }, index=index)


# In[25]:


if button :
    result=clf.predict(df)
    if result[0]==0:
        st.write("Setosa")
    elif result[0]==1:
        st.write("Versicolor")
    else:
        st.write("Virginica")
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)

    st.write(f'Accuracy =', acc)

    st.write("Classification Report")

    from sklearn.metrics import classification_report

    report_string = classification_report(y_test, y_pred)
    report_lines = report_string.split('\n')
    data = []
    for line in report_lines[2:-3]:  # Skip the first and last lines
        row_data = line.split()
        if len(row_data) >= 5:
            row_name = row_data[0]
            row_values = [float(x) for x in row_data[1:]]
            data.append([row_name] + row_values)

    # Create a DataFrame
    columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    df_report = pd.DataFrame(data, columns=columns)

    # Set the 'Class' column as the index
    df_report.set_index('Class', inplace=True)

    # Optionally, you can round the values for better presentation
    df_report = df_report.round(2)
    df_report
        
    


# In[ ]:




