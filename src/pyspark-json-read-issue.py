from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("ECommerceInsights").getOrCreate()

# Input Data File Path :
orders_single_line_json_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\orders_single_line_json.json"
orders_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\orders.json"
products_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\products.csv"

# Load orders data (JSON)
orders_df = spark.read.json(orders_single_line_json_file_path)
orders_df = spark.read.option("multiline", "true").json(orders_file_path)

# Understand the schema
orders_df.printSchema()

# Load products data (CSV)
products_df = spark.read.csv(products_file_path, header=True, inferSchema=True)

# Show loaded data
print("Show the data : ")
orders_df.show(truncate=False)
products_df.show(truncate=False)


