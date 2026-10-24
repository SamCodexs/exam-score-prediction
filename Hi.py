print("HELLO USER")
print("WELCOME TO MY STUDENT PERFORMANCE PREDICTION PROJECT!")
student_name="Samriddhi"
print("PROJECT CREATED BY:",student_name)
import pandas as pd #to avoid writing pandas everytime
#pandas is a library used to organize and analyse data
#ourproject is using pd to make table
from sklearn.linear_model import LinearRegression
#LinearRegression is a specific model we r using to learn relation between score and hours
from sklearn.model_selection import train_test_split
name=input("ENTER YOUR NAME")
hours= input("ENETR THE NUMBER OF HOURS YOU STUDIED",)
hours= float(hours)
print("Study hours entered:",hours)
data={
    "Hours": [1,2,3,4,5],
    "Score": [35,45,55,65,75]
    }
df= pd.DataFrame(data)#converts our data into table
print(df)
X=df[["Hours"]] #we use double brackett to keep x as a table with one column
Y=df["Score"] #select score column as series
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
print("INPUT X:")
print(X)
print("TARGET Y:")
print(Y)
model= LinearRegression()
model.fit(X_train,Y_train) #.fit learns the relation between study hours and score
print(model)
predicted_score=model.predict(pd.DataFrame({"Hours":[hours]}))#1 inner brack is one column while 2 inner brac is two column
print("Predicted Exam Score:",predicted_score[0])
training_predictions=model.predict(X)
print("Training predictions:",training_predictions)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(Y,training_predictions)
print("MEAN ABSOLUTE ERROR:",mae)
test_predictions=model.predict(X_test)
print("ACTUAL TEST SCORE:",list(Y_test))
print("Predicted test scores:",test_predictions)
test_mae=mean_absolute_error(Y_test,test_predictions)
print("TEST MEAN ABSOLUTE ERROR:",test_mae)