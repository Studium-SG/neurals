package sg.studium.neurals.dataset

import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import java.io.InputStream

/**
 * @param csvIn input stream producer, as we need to traverse the csv two times. Streams will be closed once done.
 * @param labelFeatureName e.g. "class"
 * @param batchSize used later for training
 */
class CsvDataSetIterator(
        val csvIn: () -> InputStream,
        labelFeatureName: String,
        val batchSize: Int
) : DsIterator {

    private val labels: IntRange
    private val xWidth: Int
    private val yWidth: Int

    private val parserSettings = CsvParserSettings()
    private val parser = CsvParser(parserSettings)

    private lateinit var header: Array<String>
    private var nextRecord: Array<String>? = null

    private var _preProcessor: DataSetPreProcessor? = null

    init {
        reset()
        labels = detectLabels(header, labelFeatureName)
        yWidth = (labels.last - labels.first + 1)
        xWidth = header.size - yWidth
    }

    private val yCols by lazy { header.slice(labels).toTypedArray() }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun getAllColumns(): Array<String> {
        return header
    }

    override fun getlYColumns(): Array<String> {
        return yCols
    }

    override fun getLabels(): MutableList<String> {
        throw java.lang.UnsupportedOperationException()
    }

    override fun cursor(): Int {
        throw UnsupportedOperationException()
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    override fun inputColumns(): Int {
        return xWidth
    }

    override fun numExamples(): Int {
        throw UnsupportedOperationException()
    }

    override fun batch(): Int {
        return batchSize
    }

    override fun next(num: Int): DataSet {

        val lines = ArrayList<Array<String>>(num)
        var toRead = num
        var skippedNullRows = 0
        var totalRows = 0
        while (nextRecord != null && toRead > 0) {
            if (nextRecord!!.none {
                @Suppress("SENSELESS_COMPARISON")
                it == null
            }) {
                lines.add(nextRecord!!)
                toRead--
            } else {
                skippedNullRows++
            }
            totalRows++
            nextRecord = parser.parseNext()
            if (nextRecord == null) parser.stopParsing()
        }

        if (skippedNullRows > 0) Logger.LOG.warn("{} rows skipped from csv file due to null values ({}%)",
                skippedNullRows, (skippedNullRows * 100) / totalRows)

        val avail = lines.size
        val x = Nd4j.create(avail, xWidth)
        val y = Nd4j.create(avail, yWidth)
        for (i in 0..avail - 1) {
            val row = lines[i]
            x.putRow(i, Nd4j.create(DoubleArray(xWidth, {
                if (it < labels.first)
                    row[it].toDouble()
                else
                    row[it + yWidth].toDouble()
            })))
            y.putRow(i, Nd4j.create(DoubleArray(yWidth, { row[labels.first + it].toDouble() })))
        }
        val ds = DataSet(x, y, null, null)
        if (_preProcessor != null) {
            _preProcessor!!.preProcess(ds)
        }
        return ds
    }

    override fun next(): DataSet {
        return next(batchSize)
    }

    override fun totalOutcomes(): Int {
        return yWidth
    }

    override fun totalExamples(): Int {
        throw UnsupportedOperationException()
    }

    override fun reset() {
        parser.beginParsing(csvIn.invoke())
        header = parser.parseNext()
        nextRecord = parser.parseNext()
    }

    override fun hasNext(): Boolean {
        return nextRecord != null
    }

    override fun asyncSupported(): Boolean {
        return false
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor?) {
        _preProcessor = preProcessor
    }

    override fun getPreProcessor(): DataSetPreProcessor? {
        return _preProcessor
    }
}