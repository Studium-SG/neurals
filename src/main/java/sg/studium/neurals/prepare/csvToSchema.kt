package sg.studium.neurals.prepare

import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import org.datavec.api.transform.schema.Schema
import java.io.InputStream


/**
 * Categorical values are sorted alphabetically.
 *
 * Skipping null values.
 *
 * @param csv input stream, will be closed once done.
 * @param categoricalLimit limit of how many distinct values can be considered as categorical variable
 * @return 'guessed' schema
 */
internal fun csvToSchema(csv: InputStream, categoricalLimit: Int = 5000): Schema {

    val settings = CsvParserSettings()
    val parser = CsvParser(settings)

    // collect values to guess type
    parser.beginParsing(csv)
    val header = parser.parseNext() ?: throw IllegalArgumentException("Empty csv file")
    val collectors = Array<MutableSet<String>>(header.size, { HashSet<String>() })

    var row = parser.parseNext()
    while (row != null) {
        (0..row.size - 1)
                .filter {
                    // limit how many values can still be considered categorical
                    collectors[it].size < categoricalLimit
                    // skip nulls
                    && row[it] != null
                }
                .forEach { collectors[it].add(row[it]) }
        row = parser.parseNext()
    }
    parser.stopParsing()

    val sb = Schema.Builder()
    for (i in 0..header.size - 1) {
        when {
            collectors[i].all { it.toDoubleOrNull() != null } -> {
                if (collectors[i].all { it.toIntOrNull() != null }) sb.addColumnInteger(header[i])
                else if (collectors[i].all { it.toLongOrNull() != null }) sb.addColumnLong(header[i])
                else sb.addColumnDouble(header[i])
            }
            collectors[i].size < categoricalLimit -> sb.addColumnCategorical(header[i],
                    // sorty by values
                    collectors[i].sorted().toList())
        // if more than collectLimit, cannot be categorical
            else -> sb.addColumnString(header[i])
        }
    }
    return sb.build()
}
