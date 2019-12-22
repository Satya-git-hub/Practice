#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('git clone http://github.com/wchill/HMP_Dataset.git')


# In[3]:


ls HMP_Dataset


# In[5]:



ls HMP_Dataset/Brush_teeth


# In[4]:


import os 
files = os.listdir('HMP_Dataset')
files = [s for s in files if '_' in s]
#files


# In[5]:


from pyspark.sql.types import StructField, StructType, IntegerType
schema = StructType(
                    [
                        StructField('x', IntegerType(), True),
                        StructField('y', IntegerType(), True),
                        StructField('z', IntegerType(), True)
                    ]
                    )


# In[6]:


df = None
from pyspark.sql.functions import lit
for category in files:
    data_files = os.listdir('HMP_Dataset/'+category)
    for data_file in data_files:
        #print(data_file)
        temp_df = spark.read.option('header', 'false').option('delimiter', ' ').csv('HMP_Dataset/'+category+'/'+data_file, schema = schema)
        temp_df = temp_df.withColumn('class', lit(category))
        temp_df = temp_df.withColumn('source', lit(data_file))
        
        if df is None:
            df = temp_df
        else : 
            df = df.union(temp_df)
        
        
df.show()


# In[7]:


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='class', outputCol='classIndex')
indexed = indexer.fit(df).transform(df)
indexed.show()


# In[10]:



from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCol='classIndex', outputCol='categoryVec')
encoded = encoder.transform(indexed)
encoded.show()


# In[11]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler 
vectorassembler = VectorAssembler(
							inputCols = ['x','y', 'z'],
							outputCol = 'features'
							)
features_vectorized = vectorassembler.transform(encoded)
features_vectorized.show()


# In[12]:


from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol = 'features', outputCol='features_norm', p=1.0)
normalized_data = normalizer.transform(features_vectorized)
normalized_data.show()


# In[17]:


from pyspark.ml import Pipeline
pipeline = Pipeline(stages = [
					indexer, encoder, vectorassembler, normalizer
					])
					
					
model = pipeline.fit(df)					
prediction = model.transform(df)
prediction.show()


# In[23]:


df_train = prediction.drop('x').drop('y').drop('z').drop('class').drop('source').drop('features')

df_train.show()

