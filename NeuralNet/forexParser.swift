import Foundation

private class CSV {
    var headers: [String] = []
    var rows: [Dictionary<String, String>] = []
    var columns = Dictionary<String, [String]>()
    var delimiter = NSCharacterSet(charactersInString: ",")

    init?(content: String?, delimiter: NSCharacterSet, encoding: UInt, error: NSErrorPointer){
        if let csvStringToParse = content{
            self.delimiter = delimiter

            let newline = NSCharacterSet.newlineCharacterSet()
            var lines: [String] = []
            csvStringToParse.stringByTrimmingCharactersInSet(newline).enumerateLines { line, stop in lines.append(line) }

            self.headers = self.parseHeaders(fromLines: lines)
            self.rows = self.parseRows(fromLines: lines)
            self.columns = self.parseColumns(fromLines: lines)
        }
    }

    convenience init?(contentsOfURL url: NSURL, delimiter: NSCharacterSet, encoding: UInt, error: NSErrorPointer) {
        let csvString = String(contentsOfURL: url, encoding: encoding, error: error);
        self.init(content: csvString,delimiter:delimiter, encoding:encoding, error: error)
    }

    convenience init?(contentsOfURL url: NSURL, error: NSErrorPointer) {
        let comma = NSCharacterSet(charactersInString: ",")
        self.init(contentsOfURL: url, delimiter: comma, encoding: NSUTF8StringEncoding, error: error)
    }

    convenience init?(contentsOfURL url: NSURL, encoding: UInt, error: NSErrorPointer) {
        let comma = NSCharacterSet(charactersInString: ",")
        self.init(contentsOfURL: url, delimiter: comma, encoding: encoding, error: error)
    }

    func parseHeaders(fromLines lines: [String]) -> [String] {
        return lines[0].componentsSeparatedByCharactersInSet(self.delimiter)
    }

    func parseRows(fromLines lines: [String]) -> [Dictionary<String, String>] {
        var rows: [Dictionary<String, String>] = []

        for (lineNumber, line) in lines.enumerate() {
            if lineNumber == 0 {
                continue
            }

            var row = Dictionary<String, String>()
            let values = line.componentsSeparatedByCharactersInSet(self.delimiter)
            for (index, header) in self.headers.enumerate() {
                if index < values.count {
                    row[header] = values[index]
                } else {
                    row[header] = ""
                }
            }
            rows.append(row)
        }

        return rows
    }

    func parseColumns(fromLines lines: [String]) -> Dictionary<String, [String]> {
        var columns = Dictionary<String, [String]>()

        for header in self.headers {
            let column = self.rows.map { row in row[header] != nil ? row[header]! : "" }
            columns[header] = column
        }

        return columns
    }
}

private extension String {
	var doubleValue: Double {
		return (self as NSString).doubleValue
	}
}

public func getForexData(path: String) -> [[String]] {
	let file = try! String(contentsOfFile: path)
	let raw = CSwiftV(String: file)
	
	return raw.rows
}



