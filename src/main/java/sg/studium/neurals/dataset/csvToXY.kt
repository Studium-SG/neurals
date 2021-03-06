package sg.studium.neurals.dataset

import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import org.slf4j.LoggerFactory
import java.io.InputStream

/**
 * This implementation does not require a header as labels specified by indices.
 *
 * @param csvIn one-hot csv, all values will be read as double. Stream will be closed once done.
 * @param labelColsFrom index of first label column, 0 based
 * @param labelColsTo index of last label column, 0 based
 * @param header if the csv file has a header row to be ignored
 * @return [XY]
 */
fun csvToXY(csvIn: InputStream, labelColsFrom: Int, labelColsTo: Int, header: Boolean = true): XY {
    val parserSettings = CsvParserSettings()
    parserSettings.maxColumns = 2048
    val parser = CsvParser(parserSettings)

    parser.beginParsing(csvIn)

    var row = parser.parseNext()
    var headerRow = emptyArray<String>()
    if (header && row != null) {
        headerRow = row
        row = parser.parseNext()
    }

    val X = ArrayList<DoubleArray>()
    val Y = ArrayList<DoubleArray>()
    val widthY = labelColsTo - labelColsFrom + 1
    val widthX = row.size - widthY

    var skippedNullRows = 0
    var totalRows = 0

    while (row != null) {
        // only if all values provided, skipping rows with any null fields
        if (row.none { it == null }) {
        X.add(kotlin.DoubleArray(widthX, {
            if (it < labelColsFrom)
                row[it].toDouble()
            else
                row[it + widthY].toDouble()
        }))
        Y.add(kotlin.DoubleArray(widthY, { row[it + labelColsFrom].toDouble() }))
        } else {
            skippedNullRows++
        }

        totalRows++
        row = parser.parseNext()
    }
    parser.stopParsing()

    if (skippedNullRows > 0) Logger.LOG.warn("{} rows skipped from csv file due to null values ({}%)",
            skippedNullRows, (skippedNullRows * 100) / totalRows)

    return XY(X, Y,
            headerRow,
            if (headerRow.isEmpty()) headerRow else headerRow.slice(IntRange(labelColsFrom, labelColsTo)).toTypedArray())
}

/**
 * This implementation requires a header.
 *
 * Compares column names with [labelFeatureName] and works with both "feature`[value`]" (aka one-hot) and simple "feature" type headers.
 *
 * Label columns have to be subsequent columns.
 *
 * @param csvIn one-hot csv, all values will be read as double. Stream will be closed once done.
 * @param labelFeatureName will be compared as "labelFeatureName[" prefix or with exact match, should not include the "["
 * @return [XY]
 */
fun csvToXY(csvIn: InputStream, labelFeatureName: String): XY {
    val parserSettings = CsvParserSettings()
    parserSettings.maxColumns = 2048
    val parser = CsvParser(parserSettings)

    parser.beginParsing(csvIn)

    val header = parser.parseNext() ?: throw RuntimeException("Empty csv file")
    val labels = detectLabels(header, labelFeatureName)
    var row = parser.parseNext()

    val X = ArrayList<DoubleArray>()
    val Y = ArrayList<DoubleArray>()
    val widthY = labels.last - labels.first + 1
    val widthX = row.size - widthY

    var skippedNullRows = 0
    var totalRows = 0

    while (row != null) {
        // only if all values provided, skipping rows with any null fields
        if (row.none { it == null }) {
            X.add(kotlin.DoubleArray(widthX, {
                if (it < labels.first)
                    row[it].toDouble()
                else
                    row[it + widthY].toDouble()
            }))
            Y.add(kotlin.DoubleArray(widthY, { row[it + labels.first].toDouble() }))
        } else {
            skippedNullRows++
        }

        totalRows++
        row = parser.parseNext()
    }
    parser.stopParsing()

    if (skippedNullRows > 0) Logger.LOG.warn("{} rows skipped from csv file due to null values ({}%)",
            skippedNullRows, (skippedNullRows * 100) / totalRows)

    return XY(X, Y, header, header.slice(labels).toTypedArray())
}

internal class Logger {
    internal companion object {
        val LOG = LoggerFactory.getLogger(Logger::class.java)!!
    }
}