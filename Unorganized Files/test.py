from sklearn.neural_network import MLPRegressor 
 
# create Trainig Dataset
train_x=[[x] for x in  range(200)]
train_y=[x[0]**2 for x in train_x]
 
#create neural net regressor
reg = MLPRegressor(hidden_layer_sizes=(50,),algorithm="l-bfgs")
reg.fit(train_x,train_y)
 
#test prediction
test_x=[[x] for x in  range(201,220,2)]
 
predict=reg.predict(test_x)
print "_Input_\t_output_"
for i in range(len(test_x)):
    print "  ",test_x[i],"---->",predict[i]
