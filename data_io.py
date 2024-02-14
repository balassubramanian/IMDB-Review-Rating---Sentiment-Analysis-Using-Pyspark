from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .getOrCreate()

def read_csv(file_path):
    return spark.read.csv(file_path, header=True, inferSchema=True)
