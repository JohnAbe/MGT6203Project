
library(dplyr)
library(tools)

## Given a list of <household_key, store_id, churn_week> return {household_key, [product_id_list]}

trans_data <-read.csv( "/Users/johnabraham/Desktop/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/transaction_data.csv")
product_list <- read.csv( "/Users/johnabraham/Desktop/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/product.csv")


## Getting a sample of a dummy output from the model
sample_model_output <-read.csv("/Users/johnabraham/Dropbox/MSARelated/Spring2018/MGT6203/ProjectRelated/final_data_use.csv")
sample_model_output <- sample_model_output[sample_model_output$if_churn==1,]
sample_model_output <- sample_model_output[sample(1:dim(sample_model_output)[1],100),]
# Now we have 100 rows with the expected output schema

get_product_lists <- function(model_output, trans=trans_data, products=product_list, purchase_history_lookbac=10, top_n_products = 2){
  names(trans) <- tolower(names(trans))
  model_output["churn_week_no"] <- model_output["week_no"]
  model_output["week_no"] <- NULL
  joined <- merge(model_output[c("household_key", "store_id", "churn_week_no")], trans, by=c("household_key", "store_id"))
  # Want to look at data prior to the predicted week of churn
  joined <- joined[joined$week_no<joined$churn_week_no,]
  # Want to summarize based on recent purchase history, default value looks back 10 weeks
  joined <- joined[joined$week_no>=(joined$churn_week_no-10),]
  # ties.method for rank by qty is "max" is because many products might just be bought once by customer
  # there might not be a clear favourite in that case and we do not want to enforce one.
  # Value based ranking is more likely to be unique but chosen max because it better than default  - "average" which gives non-integral ranks
  household_top_products <- as.data.frame(joined %>% 
                                            group_by(store_id, household_key, product_id) %>% 
                                            summarize(tot_sales_value = sum(sales_value),tot_product_qty = sum(quantity)) %>% 
                                            mutate(rank_by_value=rank(-tot_sales_value, ties.method="max"), rank_by_qty=rank(-tot_product_qty, ties.method="max")))
  # return top n products by value or by quantity
  #household_top_products <- household_top_products[which((household_top_products$rank_by_value<=top_n_products) | (household_top_products$rank_by_qty<=top_n_products)),  ]
  # For now, only ranking by value
  household_top_products <- household_top_products[which(household_top_products$rank_by_value<=top_n_products),]
  # household_top_products_recommended <- as.data.frame(household_top_products %>% 
  #                                             group_by(store_id, household_key) %>% 
  #                                               summarise(reco_prod_by_value = paste(product_id, collapse="|"), reco_prod_by_qty= paste(product_id, collapse="|")))
  
  names(products) <- tolower(names(products))

  household_reco_products <- merge(household_top_products[c("store_id","household_key","product_id")], products[c("product_id", "sub_commodity_desc")], by="product_id")
  household_reco_products$prod_name <- toTitleCase(tolower(trimws(household_reco_products$sub_commodity_desc)))
  household_reco_products <- as.data.frame(household_reco_products %>%
                                             group_by(store_id, household_key) %>%
                                             summarise(product_ids = paste(product_id, collapse="|"),
                                                       product_names = paste(prod_name, collapse="|")))
  return(household_reco_products)
}
  

# Try calling that function
head(get_product_lists(sample_model_output, trans, product_list))
