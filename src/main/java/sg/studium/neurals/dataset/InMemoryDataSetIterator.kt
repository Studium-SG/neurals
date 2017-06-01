package sg.studium.neurals.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import java.lang.UnsupportedOperationException

/**
 * In-memory iterator.
 */
class InMemoryDataSetIterator(
        private val X: List<DoubleArray>,
        private val Y: List<DoubleArray>,
        private val header: Array<String>,
        private val yCols: Array<String>,
        private val batchSize: Int
) : DsIterator {

    constructor(xy: XY, batchSize: Int) : this(xy.X, xy.Y, xy.allColumns, xy.yCols, batchSize)

    private var cursor: Int = 0
    private var gotAround: Boolean = false

    private var _preProcessor: DataSetPreProcessor? = null

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
        throw UnsupportedOperationException()
    }

    override fun cursor(): Int {
        return cursor
    }

    override fun remove() {
        throw UnsupportedOperationException()
    }

    override fun inputColumns(): Int {
        return X.first().size
    }

    override fun numExamples(): Int {
        return X.size
    }

    override fun batch(): Int {
        return batchSize
    }

    override fun next(num: Int): DataSet {
        val avail = Math.min(num, X.size - cursor)
        val x = Nd4j.create(avail, X[0].size)
        val y = Nd4j.create(avail, Y[0].size)
        for (i in 0..avail - 1) {
            x.putRow(i, Nd4j.create(X[cursor]))
            y.putRow(i, Nd4j.create(Y[cursor]))
            cursor++
            if (cursor == X.size) {
                cursor = 0
                gotAround = true
            }
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
        return Y.first().size
    }

    override fun totalExamples(): Int {
        return numExamples()
    }

    override fun reset() {
        cursor = 0
        gotAround = false
    }

    override fun hasNext(): Boolean {
        return !gotAround
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