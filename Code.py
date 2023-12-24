!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sqrt
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
spark = SparkSession.builder \
    .appName("FINALPROJECT") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "3") \
    .config("spark.executor.instances", "3") \
    .getOrCreate()
#***************************************************************************************


#*****************************************************************************************
#Read Data from CSV FIle
file_path = "/content/gdrive/MyDrive/FINALPROJECT.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
#*****************************************************************************************

#*****************************************************************************************
#Prepare the Review Summary Column for training by ENcoding Tokenizing and conveting Word2Vectors
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when

# Assuming the DataFrame has a text column named 'review_summary'
text_column = 'review_summary'

# Step 1: Filter out records with NULL values in 'review_summary' and 'rating'
df = df.filter(col(text_column).isNotNull() & col('rating').isNotNull())

# Step 2: Tokenize the text into words
tokenizer = Tokenizer(inputCol=text_column, outputCol="words")

# Step 3: Remove stopwords
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Step 4: Convert text into vectors using Word2Vec (or other encoding methods)
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="word_vectors")

# Step 5: Convert 'rating' to 'sentiment' (1 if >5, 0 otherwise)
df = df.withColumn("sentiment", when(col("rating") > 5, 1).otherwise(0))

# Create a pipeline to execute the steps
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, word2vec])

# Fit the pipeline on the DataFrame
pipeline_model = pipeline.fit(df)

# Transform the DataFrame using the fitted pipeline
df_transformed = pipeline_model.transform(df)

#*****************************************************************************************

(training_data, test_data) = df_transformed.randomSplit([0.8, 0.2], seed=123)#Making data into train and test


#*****************************************************************************************
#Build Logistic Regression Model and train on Review Summary Column
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the Logistic Regression model
lr = LogisticRegression(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lr])

# Step 3: Define the evaluator for binary classification
evaluator = BinaryClassificationEvaluator(
    labelCol="sentiment",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlr,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellr = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lr_model = cv_modellr.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lr_model.transform(test_data)

# Step 9: Calculate AUC-ROC on the test data
auc_roc = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("AUC-ROC on test data for Logistic Regression:", auc_roc)
print("Best parameter choices for Logistic Regression:")
print("RegParam for Logistic Regression:", best_lr_model.stages[-1].getRegParam())
print("ElasticNetParam for Logistic Regression:", best_lr_model.stages[-1].getElasticNetParam())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Build Linear SVC Model and train on Review Summary Column
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt

# Step 1: Define the LinearSVC model
lsvc = LinearSVC(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lsvc])

# Step 3: Define the evaluator for multiclass classification (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlsvc = ParamGridBuilder() \
    .addGrid(lsvc.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lsvc.maxIter, [10, 50, 100]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlsvc,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellsvc = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lsvc_model = cv_modellsvc.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lsvc_model.transform(test_data)

# Step 9: Calculate accuracy on the test data
accuracy = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for LinearSVC:", accuracy)
print("Best parameter choices for LinearSVC:")
print("RegParam for LinearSVC:", best_lsvc_model.stages[-1].getRegParam())
print("MaxIter for LinearSVC:", best_lsvc_model.stages[-1].getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")

results = cv_modellsvc.avgMetrics
#*****************************************************************************************

#*****************************************************************************************
#Build Random Forest Model and train on Review Summary Column
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the RandomForestClassifier model
rf = RandomForestClassifier(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[rf])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"  # You can also use "f1" for F1 score
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridrf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridrf,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelrf = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_rf_model = cv_modelrf.bestModel

# Step 8: Test the model on the test data
test_predictions = best_rf_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for RandomForestClassifier:", accuracy)
print("F1 score on test data for RandomForestClassifier:", f1_score)
print("Best parameter choices for RandomForestClassifier:")
print("NumTrees for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getNumTrees())
print("MaxDepth for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getMaxDepth())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Build Multi Layer Perceptron Model and train on Review Summary Column
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the MultilayerPerceptronClassifier model
layers = [100, 50, 2]  # contaINS 3 layers (input, hidden, output)
mlp = MultilayerPerceptronClassifier(featuresCol="word_vectors", labelCol="sentiment", layers=layers)

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[mlp])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"  # You can also use "f1" for F1 score
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridmlp = ParamGridBuilder() \
    .addGrid(mlp.blockSize, [128, 256]) \
    .addGrid(mlp.maxIter, [100, 200]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridmlp,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelmlp = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_mlp_model = cv_modelmlp.bestModel

# Step 8: Test the model on the test data
test_predictions = best_mlp_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for MultilayerPerceptronClassifier:", accuracy)
print("F1 score on test data for MultilayerPerceptronClassifier:", f1_score)
print("Best parameter choices for MultilayerPerceptronClassifier:")
print("BlockSize for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getBlockSize())
print("MaxIter for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Plot the Paramter induced accuracies for all the 4 models built on Review Summary Column
resultslr = cv_modellr.avgMetrics
reg_params = [params[lr.getParam("regParam")] for params in paramGridlr]
elastic_net_params = [params[lr.getParam("elasticNetParam")] for params in paramGridlr]

hyperparam_combinationslr = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_params, elastic_net_params)]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslr, resultslr, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Logistic Regression')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
reg_paramslsvc = [params[lsvc.getParam("regParam")] for params in paramGridlsvc]
max_iter_params = [params[lsvc.getParam("maxIter")] for params in paramGridlsvc]
resultsLSVC = cv_modellsvc.avgMetrics
min_length = min(len(reg_paramslsvc), len(max_iter_params), len(resultsLSVC))
hyperparam_combinationslsvc = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_paramslsvc[:min_length], max_iter_params[:min_length])]

# Create a line plot to visualize accuracy with different hyperparameter combinations

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslsvc, resultsLSVC, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Linear SVC')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
numtreesrf = [params[rf.getParam("numTrees")] for params in paramGridrf]
maxdepthrf = [params[rf.getParam("maxDepth")] for params in paramGridrf]
resultsrf = cv_modelrf.avgMetrics
hyperparam_combinationsrf = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(numtreesrf, maxdepthrf)]


# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsrf, resultsrf, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Random forest')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")

blockSizemlp = []
maxItermlp = []

for params in paramGridmlp:
    block_size = None
    max_iter = None
    for param in params:
        param_name = param.name
        if "blockSize" in param_name:
            block_size = params[param]
        elif "maxIter" in param_name:
            max_iter = params[param]
    if block_size is not None and max_iter is not None:
        blockSizemlp.append(block_size)
        maxItermlp.append(max_iter)

resultsmlp = cv_modelmlp.avgMetrics
hyperparam_combinationsmlp = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(blockSizemlp, maxItermlp )]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsmlp, resultsmlp, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for MLP')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()


print(" ")

models = [best_lr_model, best_lsvc_model, best_rf_model, best_mlp_model]
model_names = ["Logistic Regression", "Linear SVC", "Random Forest", "Multilayer Perceptron"]

accuracies = []
for model in models:
    test_predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(test_predictions)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1.0)
plt.show()
#*****************************************************************************************

#We will now test on the Review Detail Column

#*****************************************************************************************
#Read Data from CSV FIle
file_path = "/content/gdrive/MyDrive/FINALPROJECT.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
#*****************************************************************************************

#*****************************************************************************************
#Prepare the Review Summary Column for training by ENcoding Tokenizing and conveting Word2Vectors
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
df = df.drop("review_summary")
# DataFrame has a text column named 'review_detail'
text_column = 'review_detail'

# Step 1: Filter out records with NULL values in 'review_detail' and 'rating'
df = df.filter(col(text_column).isNotNull() & col('rating').isNotNull())

# Step 2: Tokenize the text into words
tokenizer = Tokenizer(inputCol=text_column, outputCol="words")

# Step 3: Remove stopwords
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Step 4: Convert text into vectors using Word2Vec (or other encoding methods)
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="word_vectors")

# Step 5: Convert 'rating' to 'sentiment' (1 if >5, 0 otherwise)
df = df.withColumn("sentiment", when(col("rating") > 5, 1).otherwise(0))

# Create a pipeline to execute the steps
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, word2vec])

# Fit the pipeline on the DataFrame
pipeline_model = pipeline.fit(df)

# Transform the DataFrame using the fitted pipeline
df_transformed = pipeline_model.transform(df)
#*****************************************************************************************

#*****************************************************************************************
##Build Logistic Regression Model and train on Review Detail Column
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the Logistic Regression model
lr = LogisticRegression(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lr])

# Step 3: Define the evaluator for binary classification
evaluator = BinaryClassificationEvaluator(
    labelCol="sentiment",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlr,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellr = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lr_model = cv_modellr.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lr_model.transform(test_data)

# Step 9: Calculate AUC-ROC on the test data
auc_roc = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("AUC-ROC on test data for Logistic Regression:", auc_roc)
print("Best parameter choices for Logistic Regression:")
print("RegParam for Logistic Regression:", best_lr_model.stages[-1].getRegParam())
print("ElasticNetParam for Logistic Regression:", best_lr_model.stages[-1].getElasticNetParam())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
##Build Linear SVC Model and train on Review Detail Column
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt

# Step 1: Define the LinearSVC model
lsvc = LinearSVC(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lsvc])

# Step 3: Define the evaluator for multiclass classification (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlsvc = ParamGridBuilder() \
    .addGrid(lsvc.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lsvc.maxIter, [10, 50, 100]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlsvc,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellsvc = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lsvc_model = cv_modellsvc.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lsvc_model.transform(test_data)

# Step 9: Calculate accuracy on the test data
accuracy = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for LinearSVC:", accuracy)
print("Best parameter choices for LinearSVC:")
print("RegParam for LinearSVC:", best_lsvc_model.stages[-1].getRegParam())
print("MaxIter for LinearSVC:", best_lsvc_model.stages[-1].getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")

# Step 11: Plot Accuracy for all parameters
results = cv_modellsvc.avgMetrics
#*****************************************************************************************

#*****************************************************************************************
##Build Random Forest Model and train on Review Detail Column
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the RandomForestClassifier model
rf = RandomForestClassifier(featuresCol="word_vectors", labelCol="sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[rf])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridrf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridrf,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelrf = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_rf_model = cv_modelrf.bestModel

# Step 8: Test the model on the test data
test_predictions = best_rf_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for RandomForestClassifier:", accuracy)
print("F1 score on test data for RandomForestClassifier:", f1_score)
print("Best parameter choices for RandomForestClassifier:")
print("NumTrees for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getNumTrees())
print("MaxDepth for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getMaxDepth())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
##Build Multi Layer Perceptron Model and train on Review Detail Column
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the MultilayerPerceptronClassifier model
layers = [100, 50, 2]  # contaINS 3 layers (input, hidden, output)
mlp = MultilayerPerceptronClassifier(featuresCol="word_vectors", labelCol="sentiment", layers=layers)

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[mlp])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridmlp = ParamGridBuilder() \
    .addGrid(mlp.blockSize, [128, 256]) \
    .addGrid(mlp.maxIter, [100, 200]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridmlp,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelmlp = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_mlp_model = cv_modelmlp.bestModel

# Step 8: Test the model on the test data
test_predictions = best_mlp_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for MultilayerPerceptronClassifier:", accuracy)
print("F1 score on test data for MultilayerPerceptronClassifier:", f1_score)
print("Best parameter choices for MultilayerPerceptronClassifier:")
print("BlockSize for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getBlockSize())
print("MaxIter for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Plot the Paramter induced accuracies for all the 4 models built on Review Detail Column
resultslr = cv_modellr.avgMetrics
reg_params = [params[lr.getParam("regParam")] for params in paramGridlr]
elastic_net_params = [params[lr.getParam("elasticNetParam")] for params in paramGridlr]

hyperparam_combinationslr = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_params, elastic_net_params)]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslr, resultslr, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Logistic Regression')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
reg_paramslsvc = [params[lsvc.getParam("regParam")] for params in paramGridlsvc]
max_iter_params = [params[lsvc.getParam("maxIter")] for params in paramGridlsvc]
resultsLSVC = cv_modellsvc.avgMetrics
min_length = min(len(reg_paramslsvc), len(max_iter_params), len(resultsLSVC))
hyperparam_combinationslsvc = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_paramslsvc[:min_length], max_iter_params[:min_length])]

# Create a line plot to visualize accuracy with different hyperparameter combinations

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslsvc, resultsLSVC, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Linear SVC')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
numtreesrf = [params[rf.getParam("numTrees")] for params in paramGridrf]
maxdepthrf = [params[rf.getParam("maxDepth")] for params in paramGridrf]
resultsrf = cv_modelrf.avgMetrics
hyperparam_combinationsrf = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(numtreesrf, maxdepthrf)]


# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsrf, resultsrf, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Random forest')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")

blockSizemlp = []
maxItermlp = []

for params in paramGridmlp:
    block_size = None
    max_iter = None
    for param in params:
        param_name = param.name
        if "blockSize" in param_name:
            block_size = params[param]
        elif "maxIter" in param_name:
            max_iter = params[param]
    if block_size is not None and max_iter is not None:
        blockSizemlp.append(block_size)
        maxItermlp.append(max_iter)

resultsmlp = cv_modelmlp.avgMetrics
hyperparam_combinationsmlp = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(blockSizemlp, maxItermlp )]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsmlp, resultsmlp, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for MLP')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()


print(" ")

models = [best_lr_model, best_lsvc_model, best_rf_model, best_mlp_model]
model_names = ["Logistic Regression", "Linear SVC", "Random Forest", "Multilayer Perceptron"]

accuracies = []
for model in models:
    test_predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(test_predictions)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1.0)
plt.show()
#*****************************************************************************************


#Now lets combine Review Detail and Review Summary and build a model on it
#Lets see if we can get better Accuracy from these models
#*****************************************************************************************
#Read Data from CSV FIle
file_path = "/content/gdrive/MyDrive/FINALPROJECT.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
#*****************************************************************************************
#*****************************************************************************************
#Prepare the Review Summary Column for training by ENcoding Tokenizing and conveting Word2Vectors for both Review Summary and Detail COlumns
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
summary_column = 'review_summary'
detail_column = 'review_detail'

# Step 1: Filter out records with NULL values in 'Review_Summary', 'Review_Detail', and 'Rating'
df = df.filter(col(summary_column).isNotNull() & col(detail_column).isNotNull() & col('Rating').isNotNull())

# Step 2: Tokenize the text into words for both columns
tokenizer_summary = Tokenizer(inputCol=summary_column, outputCol="words_summary")
tokenizer_detail = Tokenizer(inputCol=detail_column, outputCol="words_detail")

# Step 3: Remove stopwords for both columns
stopwords_remover_summary = StopWordsRemover(inputCol="words_summary", outputCol="filtered_words_summary")
stopwords_remover_detail = StopWordsRemover(inputCol="words_detail", outputCol="filtered_words_detail")

# Step 4: Convert text into vectors using Word2Vec for both columns
word2vec_summary = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words_summary", outputCol="word_vectors_summary")
word2vec_detail = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words_detail", outputCol="word_vectors_detail")

# Step 5: Combine features using VectorAssembler
assembler = VectorAssembler(
    inputCols=["word_vectors_summary", "word_vectors_detail"],
    outputCol="combined_features"
)

# Step 6: Convert 'Rating' to 'Sentiment' (1 if >5, 0 otherwise)
df = df.withColumn("Sentiment", when(col("Rating") > 5, 1).otherwise(0))

# Create a pipeline to execute the steps
pipeline = Pipeline(stages=[tokenizer_summary, stopwords_remover_summary, word2vec_summary,
                            tokenizer_detail, stopwords_remover_detail, word2vec_detail,
                            assembler])

# Fit the pipeline on the DataFrame
pipeline_model = pipeline.fit(df)

# Transform the DataFrame using the fitted pipeline
df_transformed = pipeline_model.transform(df)
#*****************************************************************************************

#*****************************************************************************************
#Build Logistic Regression Model and train on Review Summary + Detail Column
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the Logistic Regression model
lr = LogisticRegression(featuresCol="combined_features", labelCol="Sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lr])

# Step 3: Define the evaluator for binary classification
evaluator = BinaryClassificationEvaluator(
    labelCol="Sentiment",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlr,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellr = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lr_model = cv_modellr.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lr_model.transform(test_data)

# Step 9: Calculate AUC-ROC on the test data
auc_roc = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("AUC-ROC on test data for Logistic Regression:", auc_roc)
print("Best parameter choices for Logistic Regression:")
print("RegParam for Logistic Regression:", best_lr_model.stages[-1].getRegParam())
print("ElasticNetParam for Logistic Regression:", best_lr_model.stages[-1].getElasticNetParam())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Build Linear SVC Model and train on Review Summary + Detail Column
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt

# Step 1: Define the LinearSVC model
lsvc = LinearSVC(featuresCol="combined_features", labelCol="Sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[lsvc])

# Step 3: Define the evaluator for multiclass classification (accuracy)
evaluator = MulticlassClassificationEvaluator(
    labelCol="Sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridlsvc = ParamGridBuilder() \
    .addGrid(lsvc.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(lsvc.maxIter, [10, 50, 100]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridlsvc,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modellsvc = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_lsvc_model = cv_modellsvc.bestModel

# Step 8: Test the model on the test data
test_predictions = best_lsvc_model.transform(test_data)

# Step 9: Calculate accuracy on the test data
accuracy = evaluator.evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for LinearSVC:", accuracy)
print("Best parameter choices for LinearSVC:")
print("RegParam for LinearSVC:", best_lsvc_model.stages[-1].getRegParam())
print("MaxIter for LinearSVC:", best_lsvc_model.stages[-1].getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")

# Step 11: Plot Accuracy for all parameters
results = cv_modellsvc.avgMetrics
#*****************************************************************************************

#*****************************************************************************************
#Build Random Forest Model and train on Review Summary + Detail Column
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Step 1: Define the RandomForestClassifier model
rf = RandomForestClassifier(featuresCol="combined_features", labelCol="Sentiment")

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[rf])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="Sentiment",
    predictionCol="prediction",
    metricName="accuracy"
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridrf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridrf,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelrf = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_rf_model = cv_modelrf.bestModel

# Step 8: Test the model on the test data
test_predictions = best_rf_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for RandomForestClassifier:", accuracy)
print("F1 score on test data for RandomForestClassifier:", f1_score)
print("Best parameter choices for RandomForestClassifier:")
print("NumTrees for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getNumTrees())
print("MaxDepth for RandomForestClassifier:", best_rf_model.stages[-1]._java_obj.getMaxDepth())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
##Build Multi Layer Perceptron Model and train on Review Summary + Detail Column
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Assuming you have training_data and test_data DataFrames ready

# Step 1: Define the MultilayerPerceptronClassifier model
layers = [100, 50, 2]  # Example: A neural network with 3 layers (input, hidden, output)
mlp = MultilayerPerceptronClassifier(featuresCol="combined_features", labelCol="Sentiment", layers=layers)

# Step 2: Create a pipeline
pipeline = Pipeline(stages=[mlp])

# Step 3: Define the evaluator for accuracy and F1 score
evaluator = MulticlassClassificationEvaluator(
    labelCol="Sentiment",
    predictionCol="prediction",
    metricName="accuracy"  # You can also use "f1" for F1 score
)

# Step 4: Define the parameter grid for hyperparameter tuning
paramGridmlp = ParamGridBuilder() \
    .addGrid(mlp.blockSize, [128, 256]) \
    .addGrid(mlp.maxIter, [100, 200]) \
    .build()

# Step 5: Create a cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGridmlp,
    evaluator=evaluator,
    numFolds=10,  # Number of cross-validation folds
    seed=123  # Optional seed for reproducibility
)

# Step 6: Train the model using cross-validation
cv_modelmlp = crossval.fit(training_data)

# Step 7: Get the best model from cross-validation
best_mlp_model = cv_modelmlp.bestModel

# Step 8: Test the model on the test data
test_predictions = best_mlp_model.transform(test_data)

# Step 9: Calculate accuracy and F1 score on the test data
accuracy = evaluator.evaluate(test_predictions)
f1_score = evaluator.setMetricName("f1").evaluate(test_predictions)
print()
print("--------------------------------------------------------------------------------------------------")
# Step 10: Report results and best parameter choices
print("Accuracy on test data for MultilayerPerceptronClassifier:", accuracy)
print("F1 score on test data for MultilayerPerceptronClassifier:", f1_score)
print("Best parameter choices for MultilayerPerceptronClassifier:")
print("BlockSize for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getBlockSize())
print("MaxIter for MultilayerPerceptronClassifier:", best_mlp_model.stages[-1]._java_obj.getMaxIter())
print()
print("--------------------------------------------------------------------------------------------------")
#*****************************************************************************************

#*****************************************************************************************
#Plot the Paramter induced accuracies for all the 4 models built on Review Summary + Detail Column
resultslr = cv_modellr.avgMetrics
reg_params = [params[lr.getParam("regParam")] for params in paramGridlr]
elastic_net_params = [params[lr.getParam("elasticNetParam")] for params in paramGridlr]

hyperparam_combinationslr = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_params, elastic_net_params)]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslr, resultslr, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Logistic Regression')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
reg_paramslsvc = [params[lsvc.getParam("regParam")] for params in paramGridlsvc]
max_iter_params = [params[lsvc.getParam("maxIter")] for params in paramGridlsvc]
resultsLSVC = cv_modellsvc.avgMetrics
min_length = min(len(reg_paramslsvc), len(max_iter_params), len(resultsLSVC))
hyperparam_combinationslsvc = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(reg_paramslsvc[:min_length], max_iter_params[:min_length])]

# Create a line plot to visualize accuracy with different hyperparameter combinations

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationslsvc, resultsLSVC, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Linear SVC')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")
numtreesrf = [params[rf.getParam("numTrees")] for params in paramGridrf]
maxdepthrf = [params[rf.getParam("maxDepth")] for params in paramGridrf]
resultsrf = cv_modelrf.avgMetrics
hyperparam_combinationsrf = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(numtreesrf, maxdepthrf)]


# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsrf, resultsrf, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for Random forest')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

print(" ")

blockSizemlp = []
maxItermlp = []

for params in paramGridmlp:
    block_size = None
    max_iter = None
    for param in params:
        param_name = param.name
        if "blockSize" in param_name:
            block_size = params[param]
        elif "maxIter" in param_name:
            max_iter = params[param]
    if block_size is not None and max_iter is not None:
        blockSizemlp.append(block_size)
        maxItermlp.append(max_iter)

resultsmlp = cv_modelmlp.avgMetrics
hyperparam_combinationsmlp = [f"RegParam: {reg}, ElasticNetParam: {elastic}" for reg, elastic in zip(blockSizemlp, maxItermlp )]

# Create a line plot to visualize accuracy with different hyperparameter combinations
plt.figure(figsize=(12, 6))
plt.plot(hyperparam_combinationsmlp, resultsmlp, marker='o')
plt.xlabel('Hyperparameter Combinations')
plt.ylabel('Accuracy')
plt.title('Accuracy with Different Hyperparameter Combinations for MLP')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()


print(" ")

models = [best_lr_model, best_lsvc_model, best_rf_model, best_mlp_model]
model_names = ["Logistic Regression", "Linear SVC", "Random Forest", "Multilayer Perceptron"]

accuracies = []
for model in models:
    test_predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(test_predictions)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1.0)
plt.show()
#*****************************************************************************************

#*****************************************************************************************
##HELPFUL means Upvoted Reviews
##Non Helpful means downvoted Reviews
#Show Net Helpful Votes
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, sum
reviews_df = df.withColumn("helpful_votes", split(col("helpful_str"), ",").getItem(0).cast("int"))
reviews_df = reviews_df.withColumn("non_helpful_votes", split(col("helpful_str"), ",").getItem(1).cast("int"))

# Calculate the total helpful and non-helpful votes for each reviewer
total_votes_df = reviews_df.groupBy("reviewer").agg(sum("helpful_votes").alias("total_helpful_votes"),
                                                     sum("non_helpful_votes").alias("total_non_helpful_votes"))

# Calculate the net helpful votes (helpful - non-helpful) for each reviewer
net_helpful_votes_df = total_votes_df.withColumn("net_helpful_votes", col("total_helpful_votes") - col("total_non_helpful_votes"))

# Show the result
net_helpful_votes_df.show()
#*****************************************************************************************
#People with Highest Upvotes and Highest Downvotes
net_helpful_votes_df1 = net_helpful_votes_df.filter(col("net_helpful_votes").isNotNull())
sorted_df_desc = net_helpful_votes_df1.sort(col("net_helpful_votes").desc())
print("Reviewers with the Most Positive Votes")
# Show the sorted result
sorted_df_desc.show()
sorted_df_asc = net_helpful_votes_df1.sort(col("net_helpful_votes").asc())
print("Reviewers with the Most Negative Votes")
# Show the sorted result
sorted_df_asc.show()
#*****************************************************************************************
#Calculate the helpful_rate(which is net_helpful votes/ no of reviews)

from pyspark.sql.functions import col, split, sum, count
review_count_df = reviews_df.groupBy("reviewer").agg(count("*").alias("review_count"))

# Join the 'net_helpful_votes_df' with 'review_count_df' to get the number of reviews for each reviewer
combined_df = net_helpful_votes_df.join(review_count_df, "reviewer")

# Calculate the rate by dividing 'net_helpful_votes' by 'review_count'
combined_df = combined_df.withColumn("helpful_rate", col("net_helpful_votes") / col("review_count"))

# Show the result
combined_df.show()
#*****************************************************************************************
#Reviewers with the most Helpful Rate and the most Non-Helpful rate
#*****************************************************************************************
net_helpful_votes_df2 = combined_df.filter(col("net_helpful_votes").isNotNull())
sorted_df_desc1 = net_helpful_votes_df2.sort(col("helpful_rate").desc())
print("Reviewers with the Most Positive Votes Rate")
# Show the sorted result
sorted_df_desc1.show()
sorted_df_asc1 = net_helpful_votes_df2.sort(col("net_helpful_votes").asc())
print("Reviewers with the Most Negative Votes Rate")
# Show the sorted result
sorted_df_asc1.show()
#*****************************************************************************************
