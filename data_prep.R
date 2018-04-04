setwd("E:/mgt_6203/project/data")

# reading the data
causal_data = read.csv(file = "causal_data.csv", header = T)
txn_data = read.csv(file = "transaction_data.csv", header = T)
hh_data = read.csv(file = "hh_demographic.csv", header = T)
income_map = read.csv(file = "income_level.csv", header = T)

# threshold value for churn
thresh_churn = 360

store_data = subset(causal_data, display!= "A")
store_data$display = as.numeric(store_data$display)
store_data = subset(store_data, display > 0)
store_unique = sqldf("select STORE_ID as store_id, count(distinct PRODUCT_ID) as num_products 
                     from store_data group by store_id")
#quantile(store_unique$num_products)
store_unique <- within(store_unique, quartile <- cut(num_products, 
                                             quantile(num_products, probs=0:4/4), 
                                             include.lowest=TRUE, labels=FALSE))

names(txn_data) <- tolower(names(txn_data))
names(hh_data) <- tolower(names(hh_data))

temp_store_hh_data = sqldf("select household_key, store_id, 
                           count(distinct basket_id) as num_txns, 
                           sum(sales_value) as total_sales, 
                           sum(quantity) as total_qty, 
                           sum(retail_disc) as total_discount, 
                           max(day) as last_txn_day 
                           from txn_data
                           group by household_key, store_id")

temp_store_hh_data$ats = temp_store_hh_data$total_sales / temp_store_hh_data$num_txns
temp_store_hh_data$ads = temp_store_hh_data$total_discount / temp_store_hh_data$num_txns

combined_data = sqldf("select a.*, 
                      b.quartile, c.income_desc as income, c.household_size_desc as hh_size,
                      from temp_store_hh_data as a 
                      left join store_unique as b on a.store_id = b.store_id
                      left join hh_data as c on c.household_key = a.household_key")


filtered_data = na.omit(combined_data)
filtered_data = sqldf("select a.*, b.final_level as income_level from filtered_data as a left join 
                      income_map as b on a.income = b.income")

filtered_data$if_churn = ifelse(filtered_data$last_txn_day < thresh_churn, 1, 0)
write.csv(file = "filtered_data_final.csv", filtered_data, row.names = F)
