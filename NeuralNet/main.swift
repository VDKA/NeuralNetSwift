#!/usr/bin/env xcrun swift

import Foundation

extension Int {
	static func random(range: Range<Int>) -> Int {
		return Int(arc4random_uniform(UInt32(range.endIndex - range.startIndex))) + range.startIndex
	}
}

class TrainingData {
	
	func getData(source: DataSource) -> ([Double], [Double]){
		switch source {
		case .XOR: return getInputsTargetsXOR()
		case .Multiplication: return getInputsTargetsMult()
		case .AND: return getInputsTargetsAND()
		default: abort()
		}
	}
	
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
	
	func getLinearEquation() -> ([Double], [Double]) {
		let a = Double(Int.random(1...15))
		let b = Double(Int.random(1...15))
		let x = Double(Int(a) * Int(b))
		return ([a, b], [x])
	}
}

enum DataSource {
	case XOR
	case AND
	case Multiplication
	case Division
	case Addition
	case Subtraction
}

func runGenData(kind: DataSource, n: Int = 1000, net: Net) {
	
    let trainingData = TrainingData()
	
    for i in 1...n {
    	let inputsTargets = trainingData.getData(kind)
    	
    	net.feedForward(inputsTargets.0)
    	
    	let results = net.getResults()
    	
    	net.backProp(inputsTargets.1)
    	
    	let df = "%.2f"
    	let a = String(format: df, inputsTargets.0[0])
    	let b = String(format: df, inputsTargets.0[1])
    	let t = String(format: df, inputsTargets.1[0])
    	let r = String(format: df, results[0])
    	let rae = String(format: df, net.recentAverageError)
    	print("pass: \(i)\n\(a) \(b) -> \(t)\nRAE: \(rae) -> \(r)\n")
    }

}

func runAUSUSDData() {
	
    let manager = NSFileManager.defaultManager()

    let currentPath = manager.currentDirectoryPath

    //[high, low, open, close]
    let candles: [[String]]!
    if manager.fileExistsAtPath(currentPath + "/data.csv") {
    	candles = getForexData(currentPath + "/data.csv")
    } else {
    	abort()
    }

    let nCandlesIn = 5

    //will predict tomorrows candle.
    let net = Net(layout: [nCandlesIn * 4, nCandlesIn * 5, 4])

    for i in 5..<candles.count {
    //	print("range is \(i-5)..<\(i)")
    	let slice = candles[i-5..<i]
    	let inputArr: [Double] = slice.flatMap { $0.map { Double($0)! } }
    	let outputArr: [Double] = candles[i].map { Double($0)! }
    	
    	net.feedForward(inputArr)
    	
    	let results = net.getResults()
    	
    	net.backProp(outputArr)
    	
    	let df = "%.4f"
    	let exp = outputArr.map { String(format: df, $0) }
    	let out = results.map { String(format: df, $0) }
    	let rae = String(format: df, net.recentAverageError)
    	
    	print("exp: \(exp)")
    	print("got: \(out)")
    	print("rae: \(rae)")
    	print("")
    }
	
}

let net = Net(layout: [2, 2, 1], weights: [[0.5, 0.5], [0.5, 0.5], [1]])

runGenData(.XOR, n: 2000, net: net)

print("Done")
