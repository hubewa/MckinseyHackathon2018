library('ggplot2')
library('datetime')

library('dplyr')

library("randomForest")

library('xgboost')

library( taRifx )

setwd("E:/KaggleComp/McKinseyHackathonJan2018")

#___________________________________________________________________________
#                           FUNCTIONS
#__________________________________________________________________________


#This function is used to fill NA values in the Employer Category 2 with 5
fillEmpCat2NA <- function(df){
  naVals = is.na(df$Employer_Category2)
  
  for(i in (1:nrow(df))){
    if(naVals[i]){
      df$Employer_Category2[i] = 5
    }
  }
  
  return(df)
}

featureEngineer <- function(df){
  df$DOB = as.Date(df$DOB, format = "%d/%m/%Y")
  df$Lead_Creation_Date = as.Date(df$Lead_Creation_Date, format = "%d/%m/%Y")
  
  df$DOBDay = as.numeric(format(df$DOB, format = "%d"))
  df$DOBMonth = as.numeric(format(df$DOB, format = "%m"))
  df$DOBYear = as.numeric(format(df$DOB, format = "%Y"))
  df$Lead_Creation_DateDay = as.numeric(format(df$DOB, format = "%d"))
  df$Lead_Creation_DateMonth = as.numeric(format(df$DOB, format = "%m"))
  df$LeadCreationDOW = weekdays(df$Lead_Creation_Date)
  df$Age = as.numeric((df$DOB - df$Lead_Creation_Date)/365)
  
  drops = c("DOB", "Lead_Creation_Date", "Employer_Code")
  df <- df[, !names(df) %in% drops]
  
  
  #Fill NA's in the Employer Category 2 column to 5s
  df <- fillEmpCat2NA(df)
  
  df$Employer_Category2 = as.factor(df$Employer_Category2)
  df$Var1 = as.factor(df$Var1)
  
  #set integers to Numeric
  df$Loan_Amount = as.numeric(df$Loan_Amount)
  df$Loan_Period = as.numeric(df$Loan_Period)
  df$EMI = as.numeric(df$EMI)
  #df$Approved = as.logical(df$Approved)
  
  return(df)
}

trainModel <- function(df){
  drops = c("ID", "City_code", "Customer_Existing_Primary_Bank_Code")
  df <- df[, !names(df) %in% drops]
  
  model <- randomForest(df$Approved ~ ., data = df, ntree = 2000)
  
  return(model)
}


#___________________________________________________________________________
#                           MAIN
#__________________________________________________________________________

train <- read.csv("train/train.csv")
test <- read.csv("test/test.csv")

summary(train)
summary(test)

train  <- featureEngineer(train)
test <- featureEngineer(test)


#Testing of certain fields to see conditions in which it's NA
#trainNATest <- subset(train, Primary_Bank_Type != "G" & Primary_Bank_Type != "P")
#trainNATest2 <- subset(train, EMI > 0 & is.na(Existing_EMI))
#trainNATest3 <- subset(train, is.na(Existing_EMI))
#trainNATest4 <- subset(train, is.na(EMI) == 0 & is.na(Interest_Rate))
#trainNATest5 <- subset(train, is.na(Loan_Period) == 0 & is.na(Loan_Amount))
#trainNATest6 <- subset(train, is.na(Existing_EMI) & is.na(Interest_Rate) == 0)


#Set Certain columns to categorical values
train$Employer_Category2 = as.factor(train$Employer_Category2)
train$Var1 = as.factor(train$Var1)

trainApproved = subset(train, train$Approved == 1)

#Train subsets
trainSubset1 <- subset(train, is.na(Interest_Rate) == 0) #Set so we don't need to use Existing EMI column to predict
trainSubset1a <- subset(trainSubset1, is.na(Existing_EMI) == 0) #Set we can use EMI to predict
removeMissingEEMI <- subset(train, is.na(Existing_EMI) == 0)
trainSubset2 <- subset(removeMissingEEMI, is.na(Loan_Amount) == 0)
trainSubset3 <- removeMissingEEMI
trainSubset4 <- subset(train, is.na(Interest_Rate) & is.na(Existing_EMI)) #All Values here are zero

nrow(trainSubset1) + nrow(trainSubset2) + nrow(trainSubset3) + nrow(trainSubset4) #check whether we subset properly

drops = c("Existing_EMI")
trainSubset1 <- trainSubset1[, !names(trainSubset1) %in% drops]

drops = c("Interest_Rate", "EMI")
trainSubset2 <- trainSubset2[, !names(trainSubset2) %in% drops]

drops = c("Interest_Rate", "EMI", "Loan_Amount", "Loan_Period")
trainSubset3 <- trainSubset3[, !names(trainSubset3) %in% drops]

#Test subsets
testSubset1 <- subset(test, is.na(Interest_Rate) == 0 & is.na(Existing_EMI)) #Set so we don't need to use Existing EMI column to predict
testSubset1a <- subset(test, is.na(Interest_Rate) == 0) #Set we can use EMI to predict
removeMissingEEMI <- subset(test, is.na(Existing_EMI) == 0)
testSubset2 <- subset(removeMissingEEMI, is.na(Loan_Amount) == 0 & is.na(Interest_Rate))
testSubset3 <- subset(removeMissingEEMI, is.na(Loan_Amount)) 
testSubset4 <- subset(test, is.na(Interest_Rate) & is.na(Existing_EMI)) #All Values here are zero

nrow(testSubset1) + nrow(testSubset2) + nrow(testSubset3) + nrow(testSubset4) #check whether we subset properly

drops = c("Existing_EMI")
testSubset1 <- testSubset1[, !names(testSubset1) %in% drops]

drops = c("Interest_Rate", "EMI")
testSubset2 <- testSubset2[, !names(testSubset2) %in% drops]

drops = c("Interest_Rate", "EMI", "Loan_Amount", "Loan_Period")
testSubset3 <- testSubset3[, !names(testSubset3) %in% drops]

write.csv(trainSubset1, "data/trainSubset1.csv", row.names = FALSE)
write.csv(trainSubset1a, "data/trainSubset1a.csv", row.names = FALSE)
write.csv(trainSubset2, "data/trainSubset2.csv", row.names = FALSE)
write.csv(trainSubset3, "data/trainSubset3.csv", row.names = FALSE)
write.csv(trainSubset4, "data/trainSubset4.csv", row.names = FALSE)

write.csv(testSubset1, "data/testSubset1.csv", row.names = FALSE)
write.csv(testSubset1a, "data/testSubset1a.csv", row.names = FALSE)
write.csv(testSubset2, "data/testSubset2.csv", row.names = FALSE)
write.csv(testSubset3, "data/testSubset3.csv", row.names = FALSE)
write.csv(testSubset4, "data/testSubset4.csv", row.names = FALSE)

write.csv(train, "data/train.csv", row.names = FALSE)
write.csv(test, "data/test.csv", row.names = FALSE)




drops = c("ID", "City_Code", "Customer_Existing_Primary_Bank_Code")
newTrainSubset1 <- trainSubset1[, !names(trainSubset1) %in% drops]

