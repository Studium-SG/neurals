package sg.studium.neurals.prepare

import org.datavec.api.transform.ColumnType
import org.datavec.api.transform.metadata.CategoricalMetaData
import java.io.InputStream

/**
 * Categorical values are sorted alphabetically.
 *
 * Skipping null values.
 *
 * @param csv input streams, will be closed once done.
 * @param categoricalLimit limit of how many distinct values can be considered as categorical variable
 * @return 'guessed' header
 */
internal fun csvToOneHotHeader(csv: InputStream, categoricalLimit: Int = 5000): Array<String> {
    val schema = csvToSchema(csv, categoricalLimit)
    val header = mutableListOf<String>()
    schema.columnMetaData.forEach {
        if (it.columnType == ColumnType.Categorical) {
            val catMd = it as CategoricalMetaData
            catMd.stateNames.forEach { value ->
                header.add("${it.name}[$value]")
            }
        } else {
            header.add(it.name)
        }
    }
    return header.toTypedArray()
}