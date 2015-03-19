

'''
Building a logistic regression model for the titanic survivorship
http://www.kaggle.com/c/titanic-gettingStarted
Nicolas Theodoric
'''
##########################################
library(ISLR)
library(MASS)


'''
Read-in data
'''
setwd("D:/Documents/kaggle/Titanic")
raw_train <- read.csv("train.csv", na.strings = c('NA', ''))
raw_test <- read.csv("test.csv", na.strings = c('NA', ''))
train_copy1 <- raw_train
View(train_copy1)

#Check for NA values in the variables
summary(train_copy1)




'''
Clean the Data
'''
#lower casing column names. Changed embarked to 'port'
names(train_copy1) <- c("id", "survived", "pclass", "name", "sex", "age", "sibsp", 
                       "parch", "ticket", "fare", "cabin", "port")
#Age is important predictor, so remove rows with age = NA
train_clean <- train_copy1[!is.na(train_copy1$age),]

#check NA's removed
summary(train_clean)

#rename columns for test data too.
test_copy <- raw_test
names(test_copy) <- c("id", "pclass", "name", "sex", "age", "sibsp", 
                        "parch", "ticket", "fare", "cabin", "port")


'''
Explore Data
'''
str(train_clean)
#basic voyage statistics

##sex distribution
sex_dist <- table(train_clean$sex)
barplot(sex_dist, main = 'Survived vs Gender', ylab = 'Num of passenger',
        xlab = 'Sex', ylim = c(0,500))

##sex and class distribution
sexclass <- table(train_clean$sex, train_clean$pclass)
barplot(table(train_clean$sex, train_clean$pclass), xlab = 'Class level',
        ylab = 'frequency', legend = rownames(sexclass), main = 'Frequency distribution of sex and class')
##^ a lot more males in the lower class, just like in the titanic movie.

#age distribution
hist(train_clean$age, main = 'age distribution', xlab = 'Age', ylab = 'Frequency')

#port embarked
barplot(table(train_clean$port), main = 'port embarked distribution', xlab = 'Port embarked',
              ylab = 'frequency', names.arg = c('Cherbourg','Queenstown','Southampton'))

#seeing different fare prices
unique(train_clean$fare)
hist(train_clean$fare, main = 'Fare price distribution', xlab = 'Price', 
     ylab = 'Frequency', breaks = 100, xaxp  = c(0, 500, 10))

#closer look
hist(train_clean$fare[train_clean$fare < 100], main = 'Fare price < $100 distribution', xlab = 'Price', 
     ylab = 'Frequency', breaks = 100, xaxp  = c(0, 100, 10))

#number of siblings/spouses associated with a passenger
#number of parent / children associated
hist(train_clean$sibsp, main = 'number of siblings / spouses associated', ylab = 'Frequency')
hist(train_clean$parch, main = 'number of parent / children associated', ylab = 'Frequency')

#########Finally, let's look at breakdown of survival

#overall survival
barplot(table(train_clean$survived), main = 'Overall survival distribution',
        names.arg = c('Died', 'Survived'))

#Survival by sex
sursex <- table(train_clean$survived, train_clean$sex)
barplot(sursex, main = 'Survival by sex',
        names.arg = c('Died', 'Survived'), legend = colnames(sursex))

#Survival by class
surclass <- table(train_clean$pclass, train_clean$survived)
barplot(surclass, main = 'Survival by class',
        names.arg = c('Died', 'Survived'), col = heat.colors(length(rownames(surclass))),
        legend = c('Upper', 'Middle', 'Lower'))

#Survival by port embarked
surport <- table(train_clean$port, train_clean$survived)
barplot(surclass, main = 'Survival by port embarked',
        names.arg = c('Died', 'Survived'), col = heat.colors(length(rownames(surport))),
        legend = c('Queenstown','Cherbourg','Southampton'))

#Survival versus fare prices
boxplot(train_clean$fare ~ train_clean$survived)
boxplot(train_clean$fare ~ train_clean$survived, ylim = c(0, 300))

#Survival versus association with siblings/spouse and parent/children
surparch <- table(train_clean$survived, train_clean$parch)
surparch.prop <- prop.table(surparch,2)
barplot(surparch)
barplot(surparch.prop)

'''
Build the model
'''

#Since we have a 'test' data set given, we'll use the whole training file for training
#i.e no cross-validation.

#Use Good ol' logistic regression

train.glm <- glm(survived ~ pclass + sex + age + sibsp + parch + port + fare,
                    family = binomial(logit), data = train_clean)
summary(train.glm)


step(train.glm)
#impute the port missing value to S since it's most common. port NA value causing error
imputed_port <- with(train_clean, impute(port, 'S'))
train_clean2 <- train_clean
train_clean2$port <- imputed_port

train.glm <- glm(survived ~ pclass + sex + age + sibsp + parch + port + fare,
                 family = binomial(logit), data = train_clean2)

#Stepwise selection to see which variables are important.
step(train.glm)

#make final model survived ~ pclass + sex + age + sibsp based on step and potential interaction
#between sex and class as we saw from descriptive stats
logit.m <- glm(survived ~ pclass + sex + pclass*sex + age + sibsp, family = binomial(logit),
               data = train_clean2)
summary(logit.m) #pclass*sex interaction significant. So are all the other predictors

'''
Crank out predictions
'''
#test data has NA values for ages and port. So let's impute missing age as median age, and 
#missing port as 'S' / SoutHampton
#get new probability predictions
imputed_age <- with(test_copy, impute(age, median))
test_copy2 <- test_copy
test_copy2$age <- imputed_age

probs = predict(logit.m, newdata=test_copy2, type = "response")
pred = ifelse(probs > 0.3, 1, 0) #lowered threshold since there is more death that survivors.


pred_file <- data.frame(PassengerId = test_copy2$id, Survived = pred)
write.csv(pred_file, file = "logitpredictThreshold03.csv", row.names = FALSE)

#Score of 0.7655, pretty low tier... 
