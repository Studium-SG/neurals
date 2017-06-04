package sg.studium.neurals.prepare

import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import com.univocity.parsers.csv.CsvWriter
import com.univocity.parsers.csv.CsvWriterSettings
import org.slf4j.LoggerFactory
import java.io.InputStream
import java.io.OutputStream

/**
 * @param csvIn InputStream provider, as we will need to traverse the CSV two times (once to get the schema). Streams will be closed once done.
 * @param csvOut stream to write the one-hot CSV, always with header. Stream will be closed once done.
 * @return header written
 */
fun csvToOneHot(csvIn: () -> InputStream, csvOut: OutputStream): Array<String> {
    csvIn.invoke().use { is1 ->
        val header = csvToOneHotHeader(is1)
        csvIn.invoke().use { is2 ->
            return csvToOneHot(is2, header, csvOut)
        }
    }
}

/**
 * This implementation converts to one-hot matching the structure of a previously created csv file.
 *
 * Useful if you need to convert a test sample that has to match the training structure.
 *
 * All streams will be closed once done.
 *
 * Throws an exception if there are new values that cannot be matched to the old file.
 *
 * @param csvIn the stream to convert
 * @param csvTemplate the previous file to match, only its header row will be read (must have header)
 * @param csvOut the converted output stream
 * @return header written
 */
fun csvToOneHot(csvIn: InputStream, csvTemplate: InputStream, csvOut: OutputStream): Array<String> {
    // read template header
    val parserSettings = CsvParserSettings()
    parserSettings.maxColumns = 2048
    val parser = CsvParser(parserSettings)

    parser.beginParsing(csvTemplate)
    val templateHeader = parser.parseNext() ?: throw RuntimeException("Empty template csv")
    parser.stopParsing()

    return csvToOneHot(csvIn, templateHeader, csvOut)
}

/**
 * Converts an input to a desired one-hot output.
 *
 * Useful if you need to convert a test sample that has to match the training structure.
 *
 * All streams will be closed once done.
 *
 * Throws an exception if there are new values that cannot be matched to the old file.
 *
 * @param csvIn the stream to convert
 * @param headerToWrite the desired output header
 * @param csvOut the converted output stream
 * @param skipNew if rows with non-mappable (=new) categorical values should be skipped, default false
 * @return header written (same as [headerToWrite])
 */
fun csvToOneHot(csvIn: InputStream, headerToWrite: Array<String>, csvOut: OutputStream, skipNew: Boolean = false): Array<String> {
    val parserSettings = CsvParserSettings()
    parserSettings.maxColumns = 2048
    val parser = CsvParser(parserSettings)

    // create mapping index
    val headerMap = HeaderMap(headerToWrite)

    val writerSettings = CsvWriterSettings()
    // Sets the character sequence to write for the values that are null.
    writerSettings.nullValue = "?"
    val writer = CsvWriter(csvOut, writerSettings)

    parser.beginParsing(csvIn)
    val inHeader = parser.parseNext() ?: throw RuntimeException("Empty template csv")

    // write the same header
    writer.writeRow(headerToWrite)

    var skippedNullRows = 0
    var skippedRowsNew = 0
    var totalRows = 0
    var row = parser.parseNext()
    while (row != null) {
        // only if all values provided, skipping rows with any null fields
        if (row.none { it == null }) {
            val outRow = Array(headerToWrite.size, { "0" })
            row.withIndex().forEach {
                val feature = inHeader[it.index]
                val simpleIdx = headerMap.simples[feature]
                if (simpleIdx != null) {
                    outRow[simpleIdx] = it.value
                } else {
                    val categoricalMap = headerMap.categorical[feature]
                    if (categoricalMap != null) {
                        val categoricalIdx = categoricalMap[it.value]
                        if (categoricalIdx != null) {
                            outRow[categoricalIdx] = "1"
                        } else {
                            if (skipNew) {
                                skippedRowsNew++
                            } else {
                                throw IllegalStateException("No mapping for $feature:${it.value}")
                            }
                        }
                    } else {
                        throw IllegalStateException("No mapping for $feature:${it.value}")
                    }
                }

            }
            writer.writeRow(outRow)
        } else {
            skippedNullRows++
        }
        totalRows++
        row = parser.parseNext()
    }

    if (skippedRowsNew > 0) Logger.LOG.warn("{} records skipped as they contained new categorical values ({}%)",
            skippedRowsNew, (skippedRowsNew * 100) / totalRows)

    parser.stopParsing()
    writer.close()
    csvOut.close()

    return headerToWrite
}

internal class Logger {
    internal companion object {
        val LOG = LoggerFactory.getLogger(Logger::class.java)!!
    }
}