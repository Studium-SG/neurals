package sg.studium.neurals.dataset

/**
 * Label columns have to be subsequent columns.
 *
 * @param labelFeatureName will be compared as "labelFeatureName[" prefix or with exact match, should not include the "["
 * @return range of label columns
 */
fun detectLabels(header: Array<String>, labelFeatureName: String): IntRange {
    val labelIdxs = header.withIndex().filter { it.value == labelFeatureName || it.value.startsWith("$labelFeatureName[") }.map { it.index }
    (1..labelIdxs.size - 1)
            .filter { labelIdxs[it - 1] != labelIdxs[it] - 1 }
            .forEach { throw RuntimeException("Not all label columns are subsequent") }
    return IntRange(labelIdxs.first(), labelIdxs.last())
}