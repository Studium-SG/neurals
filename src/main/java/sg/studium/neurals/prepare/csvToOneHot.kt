package sg.studium.neurals.prepare

import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import com.univocity.parsers.csv.CsvWriter
import com.univocity.parsers.csv.CsvWriterSettings
import org.datavec.api.transform.ColumnType
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.Text
import org.slf4j.LoggerFactory
import java.io.InputStream
import java.io.OutputStream


/**
 * @return output schema after one-hot transformation of categorical columns
 */
fun csvToOneHot(csvIn: InputStream, csvOut: OutputStream, schemaIn: Schema, writeHeaders: Boolean = true): Schema {
    val tpb = TransformProcess.Builder(schemaIn)
    schemaIn.columnMetaData.filter { it.columnType == ColumnType.Categorical }.forEach { tpb.categoricalToOneHot(it.name) }
    val tp = tpb.build()

    val parserSettings = CsvParserSettings()
    val parser = CsvParser(parserSettings)

    val writerSettings = CsvWriterSettings()
    // Sets the character sequence to write for the values that are null.
    writerSettings.nullValue = "?"
    val writer = CsvWriter(csvOut, writerSettings)

    parser.beginParsing(csvIn)
    // skip header
    parser.parseNext() ?: throw IllegalArgumentException("Empty csv file")

    if (writeHeaders)
        writer.writeRow(tp.finalSchema.columnNames as Collection<Any>?)

    var skippedNullRows = 0
    var row = parser.parseNext()
    while (row != null) {
        // only if all values provided, skipping rows with any null fields
        if (row.none { it == null }) {
            writer.writeRow(tp.execute(row.map { Text(it) }).map { it.toString() })
        } else {
            skippedNullRows++
        }
        row = parser.parseNext()
    }
    parser.stopParsing()
    writer.close()

    if (skippedNullRows > 0) Logger.LOG.warn("{} rows skipped from csv file due to null values", skippedNullRows)

    return tp.finalSchema
}

/**
 * @param csvIn InputStream provider, as we will need to traverse the CSV two times (once to get the schema)
 * @param csvOut stream to write the one-hot CSV, always with header.
 */
fun csvToOneHot(csvIn: () -> InputStream, csvOut: OutputStream) {
    csvIn.invoke().use { is1 ->
        val schema = csvToSchema(is1)
        csvIn.invoke().use { is2 ->
            csvToOneHot(is2, csvOut, schema)
        }
    }
}

internal class Logger {
    internal companion object {
        val LOG = LoggerFactory.getLogger(Logger::class.java)!!
    }
}