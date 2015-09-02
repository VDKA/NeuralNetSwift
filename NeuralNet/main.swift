//
//  main.swift
//  NeuralNet
//
//  Created by Ethan Jackwitz on 9/1/15.
//  Copyright © 2015 Ethan Jackwitz. All rights reserved.
//

import Foundation

print("Hello, World!")

class TrainingData {
	
	func getInputsTargetsMult() -> ([Double], [Double]) {
		let a = Double(rand()) / Double(RAND_MAX)
		let b = Double(rand()) / Double(RAND_MAX)
		let x = a * b
		return ([a, b], [x])
	}
	
	func getInputsTargetsXOR() -> ([Double], [Double]) {
		let a = Double(Int(2.0 * Double(rand()) / Double(RAND_MAX)))
		let b = Double(Int(2.0 * Double(rand()) / Double(RAND_MAX)))
		let x = Double(Int(a) ^ Int(b))
		return ([a, b], [x])
	}
	
	func getInputsTargetsAND() -> ([Double], [Double]) {
		let a = Double(rand() % 2)
		let b = Double(rand() % 2)
		let x = Double(Int(a) & Int(b))
		return ([a, b], [x])
	}
	
	func getLayout() -> [Int] {
		return [2, 5, 2, 2, 1]
	}
	
	init (file: String) {
		
	}
}

let trainingData = TrainingData(file: "trainingData.txt")

let net = Net(layout: trainingData.getLayout())


for i in 1...50000 {
//	print("")
//	print("pass: \(i)")
	
	let inputsTargets = trainingData.getInputsTargetsMult()
	
	net.feedForward(inputsTargets.0)
	
	let results = net.getResults()
	
	net.backProp(inputsTargets.1)
	
	let df = "%.2f"
	let a = String(format: df, inputsTargets.0[0])
	let b = String(format: df, inputsTargets.0[1])
	let t = String(format: df, inputsTargets.1[0])
	let r = String(format: df, results[0])
	let rae = String(format: df, net.recentAverageError)
	print("pass: \(i)\n\(a) \(b) -> \(t)\n          -> \(r)\nRecent Average Error: \(rae)\n")
//	print("Inputs:  \(a) \(b)")
//	print("Targets: \(t)")
//	print("Outputs: \(r)")
}

print("Done")
