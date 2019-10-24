# FeedForwardNetwork.jl
A basic implementation of a feed forward neural network in julia

This programme is used to build a simple feedforward neural network
##############################
#Activation functions are in the form: functionTransform
#The derivates of the activations are in the form: functionDeriv
###############################################
Tutorial example for solving XOR using a neuralnetwork

change your directory to FeedForwardNetwork.jl
Open the julia shell and type the following functions:


include("FeedForwardNetwork.jl")

using Main.FeedForwardNetwork # This exposes all the functions

##Note that in the future I'd like for the FeedForwardNetwork.jl module to be more separated in the fur=ture when I figure out how to include modules in modules

inputs =[[0.0, 0.0],[0.0, 1.0],[1.0,0.0],[1.0,1.0]] # All possible inputs

label = [[0.0], [1.0], [1.0], [0.0]] # All expected outputs for the respective inputs

inputify(x) = [reshape(x[i],(length(x[i]), 1)) for i=1:length(x)]

#Defining the inputify function is neccessary because the network only accepts inputs and outputs of the type Array{Float64,2}
#Hopefully in a future release multiple dispatch will be used to effectively accepts multiple types of inputs and outputs

Network = BuildNetwork([2,2,1],reluTransform,reluDeriv) 

#Build the neural network with 2 neurons in the input layer, 2 (+1 for the bias neuron) neurons for the hidden layer and 1 neuron for the output layer
#Note that all the hideen layers and the input layer will have one neuron added to what you define as a bias neuron is automatically added

Input = inputify(inputs) # Inputs are defined

Label = inputify(label) # Expected oytputs are defined



#First you define the parameters for the BackPropUnit as (TheNeuralNetwork,TheInputForTheNetwork,TheExpectedOutput)
#Then you perform gradient descent with the Network using the BackPropUnit defined in step 1 and the derivate of the cost function and the learning rate

for i=1:10000

    LUnit = BackPropUnit(Network,Input[(i%4)+1],Label[(i%4)+1])
    
    SGD(LUnit,quadDeriv,0.1)
    
end


println(infer(Network,reshape([0.0,0.0],(2,1)))) # Expected output is 0

println(infer(Network,reshape([1.0,0.0],(2,1)))) # Expected output is 1

println(infer(Network,reshape([1.0,1.0],(2,1)))) # Expected output is 0

println(infer(Network,reshape([0.0,1.0],(2,1)))) # Expected output is 1
