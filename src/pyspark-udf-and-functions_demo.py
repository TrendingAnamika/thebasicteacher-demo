from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, MapType, IntegerType
from pyspark.sql.functions import col, explode, expr, arrays_zip, map_keys, map_values, map_entries
from pyspark.sql.functions import udf, array_contains, posexplode

# Create a Spark session
spark = SparkSession.builder.appName("ECommerceInsights").getOrCreate()

# Step 1: Load Data

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
orders_df = spark.read.json(orders_single_line_json_file_path)
orders_df.show(truncate=False)

# orders_df = spark.read.json(orders_file_path) # This will throw error as it is multi line json.
orders_df = spark.read.option("multiline", "true").json(orders_file_path)
orders_df.show(truncate=False)

# Read data with custom_schema.
orders_df = spark.read.option("multiline", "true").schema(custom_schema).json(orders_file_path)
orders_df.printSchema()

# Load products data (CSV)
products_df = spark.read.csv(products_file_path, header=True, inferSchema=True)

# Show loaded data
orders_df.show(truncate=False)
products_df.show(truncate=False)


####################################################################

# Step 2: Explode Nested Data
# Task: Break down the Products and Quantities arrays into rows for easier analysis.
# Exploding the Products and Quantities arrays
# exploded_df = orders_df.withColumn("Product", explode(col("Products"))) \
#                        .withColumn("Quantity", explode(col("Quantities"))) # Since we are applying explode() separately to both "Products" and "Quantities", it causes a Cartesian product effect, leading to incorrect row duplication.

# Correct Approach
# We need to zip the Products and Quantities together before exploding them. Use arrays_zip() to ensure that each product correctly maps to its corresponding quantity.
# Zip the Products and Quantities together into a struct
orders_df = orders_df.withColumn("ProductQuantity", arrays_zip(col("Products"), col("Quantities")))

# Explode the new column so each row contains one Product-Quantity pair
exploded_df = orders_df.withColumn("ProductQuantity", explode(col("ProductQuantity")))

# Extract Product and Quantity from the struct
exploded_df = exploded_df.withColumn("Product", col("ProductQuantity.Products")) \
                         .withColumn("Quantity", col("ProductQuantity.Quantities")) \
                         .drop("ProductQuantity") # Drop this temporarily created column


# exploded_df = exploded_df.drop("Products", "Quantities") # Use if want to drop the original columns after its exploded.

# Show exploded data
exploded_df.show(truncate=False)

# Drop temporarily created column ProductQuantity from orders_df
orders_df = orders_df.drop("ProductQuantity")

####################################################################

# Step 3: Handle Map Type Data
# Task: Extract keys and values from the PriceMap column and calculate total product price.
# Extracting price for each Product from the PriceMap
exploded_with_price = exploded_df.withColumn("Price", col("PriceMap")[col("Product")])

# Show the extracted price
exploded_with_price.show(truncate=False)

####################################################################

# Step 4: Extract Key, Value from Map Type Data and Create New Columns for each.
# Task: Flatten PriceMap Column to Two Columns as Product and Price

# Flatten the PriceMap column with explode using map_keys and map_values
# This approach is incorrect because we are independently exploding map_keys(col("PriceMap")) and map_values(col("PriceMap")), which results in a Cartesian product and assigns incorrect prices to products.
flattened_price_map_df = orders_df.withColumn("Product", explode(map_keys(col("PriceMap")))) \
                                  .withColumn("Price", explode(map_values(col("PriceMap"))))

# Show the result
flattened_price_map_df.show(truncate=False)

# Correct steps are below (we can take any)
# With select option
flatten_price_map_data = orders_df.select(col("OrderID"), col("CustomerID"), col("Products"), col("Quantities"), col("PriceMap"), col("OrderDate"), explode(col("PriceMap")).alias("Product", "Price"))
flatten_price_map_data = flatten_price_map_data.drop("PriceMap") # Drop PriceMap column if no longer required.
flatten_price_map_data.show(20, False)

# With SelectExpr
flatten_price_map_data = orders_df.selectExpr("*", "explode(PriceMap) as (Product, Price)")
flatten_price_map_data.show(truncate=False)

# With withColumn (First explode Product column then in next step Price column)
flatten_price_map_data = orders_df.withColumn("Product", explode(expr("map_keys(PriceMap)")))
flatten_price_map_data.show()
flatten_price_map_data = flatten_price_map_data.withColumn("Price", col("PriceMap")[col("Product")])
flatten_price_map_data.show(truncate=False)


# Another way to extract key-value pair data from PriceMap using explode and map_entries
flatten_price_map_data = orders_df.withColumn("PriceMap_Data", explode(map_entries(col("PriceMap"))))
# Extract the key and value correctly from temporarily created column PriceMap_Data
flatten_price_map_data = flatten_price_map_data.withColumn("Product", col("PriceMap_Data.key")) \
                                               .withColumn("Price", col("PriceMap_Data.value")) \
                                               .drop("PriceMap_Data")  # Drop this temporarily created column

# Drop if any duplicate (OrderID, Product, Price) rows (in this example data there is no duplicate data/rows.)
flatten_price_map_data = flatten_price_map_data.dropDuplicates(["OrderID", "Product"])
flatten_price_map_data.show(truncate=False)

####################################################################

# Step 5: Use UDF for Custom Calculation
# Task: Define a UDF to calculate the total cost (price × quantity) for each product.

# First need to extract/explode Price and Quantity column to use as input parameter in this function : calculate_total_cost(price, quantity)
# Explode `Products` and `Quantities` together using `posexplode`
orders_df_data = (orders_df.select(
    col("OrderID"), col("CustomerID"), col("Products"), col("Quantities"),
    col("PriceMap"), col("OrderDate"),
    posexplode(col("Products")).alias("pos", "Product")  # Generates (index, Product)
).withColumn("Price", expr("PriceMap[Product]")) # Fetch correct price from PriceMap
.withColumn("Quantity", col("Quantities")[col("pos")]))  # Fetch correct Quantity

# Drop indexing column pos as it is no longer required.
orders_df_data = orders_df_data.drop("pos")
orders_df_data.show(20, False)



# Define a function for total cost calculation
def calculate_total_cost(price, quantity):
    return price * quantity if price and quantity else None

# Register the UDF
total_cost_udf = udf(calculate_total_cost, IntegerType())

# Apply the UDF
orders_with_cost = orders_df_data.withColumn("TotalCost", total_cost_udf(col("Price"), col("Quantity")))

# Show data with total cost
orders_with_cost.show(truncate=False)


# Define above Functions as UDF for total cost calculation
@udf(IntegerType())
def calculate_total_cost(price, quantity):
    return price * quantity if price is not None and quantity is not None else None

# Apply the UDF to create the TotalCost column
orders_with_cost = orders_df_data.withColumn("TotalCost", calculate_total_cost(col("Price"), col("Quantity")))

# Show data with total cost
orders_with_cost.show(truncate=False)

####################################################################
# Step 6: Join with Product Data
# Task: Join with the products_df to enrich the order data with product details.

# Join with products data
detailed_orders = orders_with_cost.join(products_df, orders_with_cost.Product == products_df.ProductID, "left")

# Get the enriched order data
detailed_orders = detailed_orders.select("OrderID", "CustomerID", "ProductName", "Category", "Quantity", "Price", "TotalCost")
detailed_orders.show(truncate=False)

####################################################################
# Step 7: Column Functions
# Task: Use built-in column functions for analysis.
# Calculate total order cost : Group by OrderID, sum TotalCost, and rename the column to OrderTotal
order_totals = detailed_orders.groupBy("OrderID").sum("TotalCost").withColumnRenamed("sum(TotalCost)", "OrderTotal")

# Filter orders with total cost > 1000
high_value_orders = order_totals.filter(col("OrderTotal") > 1000)

high_value_orders.show(truncate=False)

# Join with the original DataFrame to retain all columns
detailed_orders = detailed_orders.join(high_value_orders, on="OrderID", how="left")

# Show high-value orders
detailed_orders.show(truncate=False)

####################################################################
# Step 8: Array and Map Functions
# Task: Perform operations on arrays and maps.
# Check if a specific product is in the order (e.g., "P003"). It should populate true if value exists else false.
orders_with_desk = orders_df.withColumn("ContainsDesk", array_contains(col("Products"), "P003"))
orders_with_desk.show(truncate=False)


# Flatten the PriceMap column and get keys and values from the PriceMap
price_keys = orders_df.withColumn("ProductNames", map_keys(col("PriceMap"))) \
                      .withColumn("ProductPrices", map_values(col("PriceMap")))

price_keys.show(truncate=False)


####################################################################
# Step 9: Final Aggregation
# Task: Aggregate total sales by category.
# Aggregate total sales by category
category_sales = detailed_orders.groupBy("Category").sum("TotalCost").withColumnRenamed("sum(TotalCost)", "CategoryTotal")
detailed_orders.show(truncate=False)

# Show category-level sales
category_sales.show(truncate=False)


