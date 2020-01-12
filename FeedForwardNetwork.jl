#!/usr/local/bin/julia
module FeedForwardNetwork
import Statistics.mean
#####Functions to help build the neural network
export ConnectedLayer, NetworkArchitecture,CreateLayer,addLayer,BuildNetwork, appendBias,sigmoidTransform, sigmoidDeriv, tanhTransform, tanhDeriv, reluTransform, reluDeriv, quadCost, quadDeriv, crossEntropy, crossEntropyDeriv, forwardPass!,backwardPass!, SGD, infer, BackPropUnit

function appendBias(a::Array{Float64,2})
#Function to append 1 to the end of the input to accomodate the bias in the layer 
        vcat(a,ones(Float64,1,size(a,2))) 
end


function sigmoidTransform(parameters::Array{Float64,2},input)
        return Float64(1.0) ./ (Float64(1.0) .+ exp.(-(parameters * appendBias(input))))
end


sigmoidDeriv(x::Array{Float64,2}) = return x .*  (Float64(1.0) .- x)

tanhTransform(parameters::Array{Float64,2},input) = return tanh.(parameters * appendBias(input))

tanhDeriv(input::Array{Float64,2}) = Float64(1.0) .- input.^2

function reluTransform(parameters::Array{Float64,2},input)
	x = parameters * appendBias(input)
	return x .* (x .> Float64(0.0))
end

reluDeriv(input::Array{Float64,2}) = return convert(Array{Float64,2}, input .> Float64(0))

#From here its cost functions

function quadCost(predict,real)
	return sum(0.5 * ( predict - real).^2)
end

function quadDeriv(predict,real)
	return (predict - real)
end

#####Functions to help build the neural network
mutable struct ConnectedLayer #parameters,transform,derivative}
###struct containing all the data required to define a fully connected layer
	inputsize::Int64
	NumNeurons::Int64
	parameters::Array{Float64,2}
	transform::Function 
	derivative::Function	
end

mutable struct NeuralNetwork
#A struct containing all the data required to define a multilayer neural network
	layers::Array{ConnectedLayer}
	function NeuralNetwork(firstLayer::ConnectedLayer)
		return new([firstLayer])
	end 
end

function CreateLayer(inputsize::Int, NumNeurons::Int, transform::Function, derivative::Function)
#An initialiser for the ConnectedLayer struct
	return ConnectedLayer(inputsize,NumNeurons,rand(NumNeurons, inputsize + 1),transform,derivative) #The +1 is to take into accoint the bias of each layer
end



function addLayer(neuralnet::NeuralNetwork,NumNeurons::Int64,Transform, Derivative)
# A function which adds a single fully connected layer to a neural network defined as the parameter neuralnet
	lastLayer = neuralnet.layers[end]
	inputSize = lastLayer.NumNeurons
	sigmoidLayer = CreateLayer(inputSize,NumNeurons,Transform,Derivative)
	push!(neuralnet.layers,sigmoidLayer)
end

function BuildNetwork(sizes, Transform, derivative)
#A helper function to help build a neural network  
	firstLayer = CreateLayer(sizes[1],sizes[2], Transform, derivative)
	neuralnet = NeuralNetwork(firstLayer)
	for i in 3:(length(sizes)-1)
		addLayer(neuralnet,sizes[i],Transform,derivative)
	end
	addLayer(neuralnet,sizes[end],Transform,derivative)
	return neuralnet
end


######Functions to help train the network 
   
function infer(neuralnet::NeuralNetwork,input)
#A function which reveals what the neural network would've
	currentResult = input
	for i in 1:length(neuralnet.layers)
		currentResult = neuralnet.layers[i].transform(neuralnet.layers[i].parameters, currentResult)
	end
	return currentResult
end

struct BackPropUnit
#A struct which provides all the variables required for backpropogation
  networkArchitecture::NeuralNetwork
  dataBatch::Array{Float64,2}
  labels::Array{Float64,2}
  outputs::Array{Array{Float64,2}} # outputs remembered now
  deltas::Array{Array{Float64,2}} # deltas kept here

  function BackPropUnit(arch::NeuralNetwork, dataBatch::Array{Float64,2}, labels::Array{Float64,2})
	# An initializer for the BackPropUnit struct
     outputs = [ zeros(Float16,l.NumNeurons, 1) for l in arch.layers ] #=Used length(dataBatch) for outputs and deltas =#
     # need to understand purpose of databatch for learning
     deltas = [ zeros(Float16,l.NumNeurons, 1) for l in arch.layers ]
     return new(arch, dataBatch, labels, outputs, deltas)  

  end 
end

function forwardPass!(learningUnit::BackPropUnit)
# initialises learningUnit.outputs to the output of each layer
	CurrentPass = learningUnit.dataBatch
	for i in 1:length(learningUnit.networkArchitecture.layers)
		layer = learningUnit.networkArchitecture.layers[i]
		CurrentPass = layer.transform(layer.parameters, CurrentPass)
		learningUnit.outputs[i] = CurrentPass
	end
end

function backwardPass!(learningUnit::BackPropUnit,costderiv::Function)
# initialises learningUnit.deltas to the errors of each layer
#Last layer uses the cost function to determine the error
        learningUnit.deltas[end] = costderiv(learningUnit.outputs[end],learningUnit.labels) .* learningUnit.networkArchitecture.layers[end].derivative(learningUnit.outputs[end]) # end layer deltas
        	
        for i in 1:(length(learningUnit.networkArchitecture.layers)-1 )
		      previousLayer = learningUnit.networkArchitecture.layers[end - i + 1]
      		currentLayer = learningUnit.networkArchitecture.layers[end - i]
      		learningUnit.deltas[end-i] = currentLayer.derivative(learningUnit.outputs[end-i]) .* (transpose(previousLayer.parameters[:,(1:end-1)]) * learningUnit.deltas[end - i + 1])
  	end
end

function SGD(lunit::BackPropUnit, costderiv::Function,lrate::AbstractFloat)
#This function uses stochastic gradient descent to update the parameters of the network
    forwardPass!(lunit) 
    backwardPass!(lunit,costderiv)
    dw = lunit.deltas[1] * transpose(lunit.dataBatch) # Input to first layer is databatch which isn't included in lunit.outputs
    lunit.networkArchitecture.layers[1].parameters[:,1:(end-1)] .-= dw*lrate
    lunit.networkArchitecture.layers[1].parameters[:,end] -= lunit.deltas[1]*lrate
    
    #update parameters with this loop
    for i=2:(length(lunit.networkArchitecture.layers)) #1st layer already updated so begin loop with the 2nd layer
       dw =  lunit.deltas[i] * transpose(lunit.outputs[i-1]) 
        # db = deltas
        Layer = lunit.networkArchitecture.layers[i] 
        Layer.parameters[:,1:(end-1)] .-= lrate*dw # Update weights
        Layer.parameters[:,end] -= lrate*lunit.deltas[i] # update biases
    end
end

end
