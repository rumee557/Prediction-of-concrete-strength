#Model-building data set
#Similar code was used to obtain random data set for new data set
rm(list=ls())
setwd("~/summer sem/Data analysis/Project")
concrete_data<- read.table("Concrete_Data.txt",sep="\t",header=TRUE)
names(concrete_data) <- c("Cement","Slag","Fly Ash","Water",
                     "Superplast","CA","FA",
                     "Age","Comp_Str")

train.rows<- createDataPartition(y= concrete_data$Comp_Str, 
                                 p=0.5, list = FALSE)
train.data<- concrete_data[train.rows,] # 50% data goes in here
names(train.data) <- c("Cement","Slag","Fly Ash","Water",
                       "Superplast","CA","FA",
                       "Age","Comp_Str")
predict.data <- concrete_data[-train.rows,]
names(predict.data) <- c("Cement","Slag","Fly Ash","Water",
                      "Superplast","CA","FA",
                      "Age","Comp_Str")

write.table(train.data,file="Concrete_train_data.txt",append = FALSE,sep="\t",
            dec=".",row.names=FALSE,col.names=FALSE)
write.table(predict.data,file="Concrete_predict_data.txt",append = FALSE,sep="\t",
            dec=".",row.names=FALSE,col.names=FALSE)