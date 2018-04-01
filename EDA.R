causal_data <- read.csv("Desktop/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/causal_data.csv",sep=",", stringsAsFactors = F)
rm(causal_data)

transaction_data <- read.csv("Desktop/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/transaction_data.csv",sep=",", stringsAsFactors = F)
head(transaction_data)

## Overall min max transaction dates in the file
min(transaction_data$DAY)
max(transaction_data$DAY)
hist(transaction_data$DAY)

library(dplyr)
tr_household_summarised <- as.data.frame(transaction_data %>% group_by(household_key) %>% summarise(max_day= max(DAY), min_day=min(DAY)))
head(tr_household_summarised)

# Range for days  - retain only household ids that have some activity
# prior to min_day and max_day
min_day_range = 180
max_day_range = 360

householdlist = tr_household_summarised[tr_household_summarised$min_day<=min_day_range & tr_household_summarised$max_day>=max_day_range,'household_key']
length(householdlist)


## Also have to add the retail_desc, coupon_disc etc..
household_tot_sales <- as.data.frame(transaction_data[transaction_data$household_key %in% householdlist, ] %>% group_by(household_key) %>% summarise(total_sales_value = sum(SALES_VALUE)))


household_data <- read.csv("Desktop/dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/hh_demographic.csv",sep=",", stringsAsFactors = F)
head(household_data)
## Ensuring that household_key is unique in this table
dim(household_data)[1]==length(unique(household_data$household_key))

## Getting household attributes to the summarized sales data
household_tot_sales = merge(household_tot_sales,household_data,by = 'household_key')
head(household_tot_sales)

## Ensuring no duplicate household_key is present
dim(household_tot_sales)[1]==length(unique(household_tot_sales$household_key))
row.names(household_tot_sales) <- household_tot_sales$household_key
head(household_tot_sales)
household_tot_sales$household_key <- NULL
household_tot_sales$HOUSEHOLD_SIZE_DESC <- factor(household_tot_sales$HOUSEHOLD_SIZE_DESC)
mod1  <- lm(total_sales_value~.,household_tot_sales)
summary(mod1)


library(randomForest)
#mod2 <- randomForest(household_tot_sales[setdiff(names(household_tot_sales),"total_sales_value")], household_tot_sales$total_sales_value)
mod2 <- randomForest(total_sales_value~.,data=household_tot_sales, na.action=na.omit,mtry=3)
for (name in names(household_tot_sales)){
  if(class(household_tot_sales[,name])!='numeric'){
    cat(paste("Field: ",name,sep=" "))
    print(unique(household_tot_sales[,name]))
    household_tot_sales[,name] <- factor(household_tot_sales[,name])
  }
    
  
}

