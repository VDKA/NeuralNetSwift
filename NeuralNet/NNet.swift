import Foundation

func randomWeight() -> Double {
	return Double(arc4random()) / Double(UInt32.max)
}

public struct Connection {
	var weight: Double
	var deltaWeight: Double = 0.0
	
	init (weight: Double) {
		self.weight = weight
		deltaWeight = 0.0
	}
}

public class Neuron {
	
	private var eta: Double = 0.15	//learning rate
	private var alpha: Double = 0.5
	private var nIndex: Int
	private var gradient = Double()
	public var outputWeights = [Connection]()
	public var outputValue: Double
	
	public init(numOutputs: Int, index: Int, weight: Double = randomWeight()) {
		outputValue = 0.0
		nIndex = index
		
		for _ in 0..<numOutputs {
			outputWeights.append(Connection(weight: weight))
		}
	}
	
	public func feedForward(previousLayer: [Neuron], cLayer: [Neuron]) {
		var sum = 0.0
		
		//including bias neurons
		for n in 0..<previousLayer.count {
			sum += previousLayer[n].outputValue * previousLayer[n].outputWeights[nIndex].weight
		}
		
    	outputValue = transferFunction(sum)
	}
	
	public func calculateOutputGradients(targetValue: Double) {
		let delta = targetValue - outputValue
		gradient = delta * transferFunctionDerivative(outputValue)
	}
	
	public func calculateHiddenGradients(nextLayer: [Neuron]) {
		let dow = sumDOW(nextLayer)
		gradient = dow * transferFunctionDerivative(outputValue)
	}
	
	public func updateInputWeights(prevLayer: [Neuron]) {
		for n in 0..<prevLayer.count {
			let neuron = prevLayer[n]
			let oldDeltaWeight = neuron.outputWeights[nIndex].deltaWeight
			let newDeltaWeight = eta * neuron.outputValue * gradient + alpha * oldDeltaWeight
			
			neuron.outputWeights[nIndex].deltaWeight = newDeltaWeight;
			neuron.outputWeights[nIndex].weight += newDeltaWeight;
		}
	}
	
	private func sumDOW(nextLayer: [Neuron]) -> Double {
		var sum = 0.0
		
		for n in 0..<nextLayer.count - 1 {
			sum += outputWeights[n].weight * nextLayer[n].gradient
		}
		
		return sum
	}
	
	private func transferFunction(x: Double) -> Double {
		return tanh(x)
	}
	
	private func transferFunctionDerivative(x: Double) -> Double {
		
		return 1.0 - x * x
		
	}
	
}

public class Net {
	private var layers = [[Neuron]]()
	private var error = 0.0
	private let recentAverageSmoothingFactor = 100.0
	
	public var recentAverageError = 0.0
	
	public init(layout: [Int], bias: Double = 1.0) {
		
		for layerNum in 0..<layout.count {
			
			let nOutputs = layerNum == layout.count - 1 ? 0 : layout[layerNum + 1]
			layers.append([])
			for neuronNum in 0...layout[layerNum] {
				let newNeuron = Neuron(numOutputs: nOutputs, index: neuronNum)
        		print("Made a Neuron!")
				
				if neuronNum == layout[layerNum] {
					//this neuron is a bias neuron.
					newNeuron.outputValue = bias
				}
				
				layers[layerNum].append(newNeuron)
			}
			print("")
		}
	}
	
	public init(layout: [Int], weights: [[Double]], bias: Double = 0.0) {
		
		assert(layout.count == weights.count, "Passed \(layout.count) layers, but only weights for \(weights.count) layers.")
		for i in 0..<layout.count {
			assert(layout[i] == weights[i].count, "Passed \(layout[i]) neurons for layer \(i), but only \(weights[i].count) weights for this layer.")
		}
		
		for layerNum in 0..<layout.count {
			//if final layer then nOutputs is 0, else nOutputs = nNeurons in next layer.
			let nOutputs = layerNum == layout.count - 1 ? 0 : layout[layerNum + 1]
			layers.append([])
			for neuronNum in 0...layout[layerNum] {
				let newNeuron = Neuron(numOutputs: nOutputs, index: neuronNum, weight: weights[layerNum][neuronNum])
				print("Made a Neuron!")
				
				if neuronNum == layout[layerNum] {
					//this neuron is a bias neuron
					newNeuron.outputValue = bias
				}
				
				layers[layerNum].append(newNeuron)
			}
			print("")
			
		}
	}
	
	public func getResults() -> [Double] {
		var results = [Double]()
		for n in 0..<layers.last!.count {
			results.append(layers.last![n].outputValue)
		}
		return results
	}
	
	public func feedForward(inputValues: [Double]) {
		assert(layers.first!.count-1 == inputValues.count, "Handed wrong number of input values!\nExpected \(layers.first!.count) got \(inputValues.count)\n")
		
		//assign the input values to input neurons
		for i in 0..<inputValues.count {
			layers[0][i].outputValue = inputValues[i]
		}
		
		//forward propogate
		for layerNum in 1..<layers.count {
			let prevLayer = layers[layerNum - 1]
			for n in 0..<layers[layerNum].count - 1 {
				layers[layerNum][n].feedForward(prevLayer, cLayer: layers[layerNum])
			}
		}
	}
	
	public func backProp(targetValues: [Double]) {
		let outputLayer = layers.last!
		var error = 0.0
		
		for i in 0..<outputLayer.count - 1{
			let delta = targetValues[i] - outputLayer[i].outputValue
			error += delta * delta
		}
		error /= Double(outputLayer.count - 1)
		error = sqrt(error)
		
		recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0)
		
		//calculate gradients on output layers
		for n in 0..<outputLayer.count - 1 {
			outputLayer[n].calculateOutputGradients(targetValues[n])
		}
		
		//calculate gradients on hidden layers
		for i in (1...layers.count-2).reverse() {
			let hiddenLayer = layers[i]
			let nextLayer = layers[i+1]
			
			for n in hiddenLayer {
				n.calculateHiddenGradients(nextLayer)
			}
		}
		
		//for all layers except input layer update the weights.
		
		for layerNum in (1...layers.count-1).reverse() {
			let layer = layers[layerNum]
			let prevLayer = layers[layerNum - 1]
			
			for n in 0..<layer.count - 1 {
				layer[n].updateInputWeights(prevLayer)
			}
		}
	}
}
