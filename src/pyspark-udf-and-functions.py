from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, IntegerType
from pyspark.sql.functions import explode, col, udf, map_keys, map_values, array_contains, map_entries
from pyspark.sql.functions import posexplode, expr


# Create a Spark session
spark = SparkSession.builder.appName("ECommerceInsights").getOrCreate()

# Input Data File Path :
orders_single_line_json_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\orders_single_line_json.json"
orders_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\orders.json"
products_file_path = "C:\\Users\\hp\\Desktop\\pyspark-project-demo\\input_data\\products.csv"

# Define schema explicitly
custom_schema = StructType([
    StructField("OrderID", StringType(), True),
    StructField("CustomerID", StringType(), True),
    StructField("Products", ArrayType(StringType()), True),
    StructField("Quantities", ArrayType(IntegerType()), True),
    StructField("PriceMap", MapType(StringType(), IntegerType()), True),
    StructField("OrderDate", StringType(), True)
])

# Load orders data (JSON)
#orders_df = spark.read.json(orders_single_line_json_file_path)
orders_df = spark.read.option("multiline", "true").schema(custom_schema).json(orders_file_path)
orders_df.printSchema()

# Load products data (CSV)
products_df = spark.read.csv(products_file_path, header=True, inferSchema=True)

# Show loaded data
orders_df.show(truncate=False)
products_df.show(truncate=False)

# Exploding PriceMap column as Product and Price and then exploding Quantities as Quantity column to use as required input parameter columns :

# Step : Explode `Products` and `Quantities` together using `posexplode`
exploded_orders_df = (orders_df.select(col("OrderID"), col("CustomerID"), col("Products"), col("Quantities"),col("PriceMap"), col("OrderDate"),
    posexplode(col("Products")).alias("pos", "Product")  # Generates (index, Product)
).withColumn("Price", expr("PriceMap[Product]")) # Fetch correct price from PriceMap
.withColumn("Quantity", col("Quantities")[col("pos")]))  # Fetch correct quantity

# Fetch correct price from PriceMap
exploded_orders_df = exploded_orders_df.drop("pos")
exploded_orders_df = exploded_orders_df.drop_duplicates(["OrderID", "Product"])

exploded_orders_df.show(20, False)


# Step 5: Use UDF for Custom Calculation
# Task: Define a UDF to calculate the total cost (price × quantity) for each product.
# Define a UDF for total cost calculation
def calculate_total_cost(price, quantity):
    return price * quantity if price and quantity else None


# Register the UDF
total_cost_udf = udf(calculate_total_cost, IntegerType())

# Apply the UDF
orders_with_cost = exploded_orders_df.withColumn("TotalCost", total_cost_udf(col("Price"), col("Quantity")))

# Show data with total cost
orders_with_cost.show(truncate=False)

##### Another way to define UDF without registering separately.

# Define a Functions as UDF for total cost calculation
@udf(IntegerType())
def calculate_total_cost(price, quantity):
    return price * quantity if price is not None and quantity is not None else None

# Apply the UDF to create the TotalCost column
orders_with_cost_new = exploded_orders_df.withColumn("TotalCost", calculate_total_cost(col("Price"), col("Quantity")))

# Show data with total cost
orders_with_cost_new.show(truncate=False)


##################
# Step 6: Join with Product Data
# Task: Join with the products_df to enrich the order data with product details.
# Join with products data
detailed_orders = orders_with_cost.join(products_df, orders_with_cost.Product == products_df.ProductID, "left")

# Get the enriched order data
detailed_orders = detailed_orders.select("OrderID", "CustomerID", "ProductName", "Category", "Quantity", "Price", "TotalCost")
detailed_orders.show(truncate=False)


# Step 7: Column Functions
# Task: Use built-in column functions for analysis.
# Calculate total order cost : Group by OrderID, sum TotalCost, and rename the column to OrderTotal
order_totals = detailed_orders.groupBy("OrderID").sum("TotalCost").withColumnRenamed("sum(TotalCost)", "OrderTotal")

# Filter orders with total cost > 1000
high_value_orders = order_totals.filter(col("OrderTotal") > 1000)

# Join with the original DataFrame to retain all columns
detailed_orders = detailed_orders.join(high_value_orders, on="OrderID", how="left")


# Show high-value orders
detailed_orders.show(truncate=False)


# Step 8: Array and Map Functions
# Task: Perform operations on arrays and maps.
# Check if a specific product is in the order (e.g., "P003")
orders_with_desk = orders_df.withColumn("ContainsDesk", array_contains(col("Products"), "P003"))
orders_with_desk.show(truncate=False)


# Flatten the PriceMap column
flattened_price_map_df = orders_df.withColumn("PriceMap_Key", explode(map_keys(col("PriceMap")))) \
                                  .withColumn("PriceMap_Value", explode(map_values(col("PriceMap"))))

# Show the result
flattened_price_map_df.show(truncate=False)

# Get keys and values from the PriceMap
price_keys = orders_df.withColumn("ProductKeys", map_keys(col("PriceMap"))) \
                      .withColumn("ProductPrices", map_values(col("PriceMap")))

price_keys.show(truncate=False)



# Explode Map Keys and Values Columns Separately using withColumn
flattened_price_map_df = orders_df.withColumn("PriceMap_Data", explode(map_entries(col("PriceMap"))))
flattened_price_map_df.show(truncate=False)
# Extract the key and value correctly
flattened_price_map_df = flattened_price_map_df.withColumn("Product_Name", col("PriceMap_Data.key")) \
                                               .withColumn("Product_Price", col("PriceMap_Data.value")) \
                                               .drop("PriceMap_Data")  # Clean up

# Drop duplicate (OrderID, product, price) rows
flattened_price_map_df = flattened_price_map_df.dropDuplicates(["OrderID", "Product_Name"])
flattened_price_map_df.show(truncate=False)

# Above Step with SelectExpr
flatten_orders_df_with_selectExpr = orders_df.selectExpr("*", "explode(PriceMap) as (Product, Price)")
flatten_orders_df_with_selectExpr.show()

#################################################################################################################

# Step 9: Final Aggregation
# Task: Aggregate total sales by category.
# Aggregate total sales by category
category_sales = detailed_orders.groupBy("Category").sum("TotalCost").withColumnRenamed("sum(TotalCost)", "CategoryTotal")
detailed_orders.show(truncate=False)
# Show category-level sales
category_sales.show(truncate=False)

