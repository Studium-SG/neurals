package sg.studium.neurals.dataset

import org.junit.Test
import resource
import kotlin.test.assertEquals

class CsvDataSetIteratorTest {

    @Test
    fun simpleLabel() {
        val iterator = CsvDataSetIterator({ resource("cpu.onehot.data.csv") }, "class", 10)
        assertEquals(
                listOf("vendor[adviser]", "vendor[amdahl]", "vendor[apollo]", "vendor[basf]", "vendor[bti]", "vendor[burroughs]", "vendor[c.r.d]", "vendor[cambex]", "vendor[cdc]", "vendor[dec]", "vendor[dg]", "vendor[formation]", "vendor[four-phase]", "vendor[gould]", "vendor[harris]", "vendor[honeywell]", "vendor[hp]", "vendor[ibm]", "vendor[ipl]", "vendor[magnuson]", "vendor[microdata]", "vendor[nas]", "vendor[ncr]", "vendor[nixdorf]", "vendor[perkin-elmer]", "vendor[prime]", "vendor[siemens]", "vendor[sperry]", "vendor[sratus]", "vendor[wang]", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "class"),
                iterator.getAllColumns().toList())
        assertEquals(
                listOf("class"),
                iterator.getlYColumns().toList())
        assertEquals(36, iterator.inputColumns())
        assertEquals(1, iterator.totalOutcomes())
        var ds = iterator.next()
        var total = ds.features.rows()
        assertEquals(36, ds.features.columns())
        assertEquals(10, ds.features.rows())
        assertEquals(1, ds.labels.columns())
        assertEquals(10, ds.labels.rows())
        while (iterator.hasNext()) {
            ds = iterator.next()
            total += ds.features.rows()
        }
        assertEquals(209, total)
        assertEquals(36, ds.features.columns())
        assertEquals(9, ds.features.rows()) // total 209, last one 9
        assertEquals(1, ds.labels.columns())
        assertEquals(9, ds.labels.rows())
    }

    @Test
    fun multiLabel() {
        val iterator = CsvDataSetIterator({ resource("iris.onehot.csv") }, "class", 10)
        assertEquals(
                listOf("sepallength", "sepalwidth", "petallength", "petalwidth", "class[Iris-setosa]", "class[Iris-versicolor]", "class[Iris-virginica]"),
                iterator.getAllColumns().toList())
        assertEquals(
                listOf("class[Iris-setosa]", "class[Iris-versicolor]", "class[Iris-virginica]"),
                iterator.getlYColumns().toList())
        assertEquals(4, iterator.inputColumns())
        assertEquals(3, iterator.totalOutcomes())
        var ds = iterator.next()
        var total = ds.features.rows()
        assertEquals(4, ds.features.columns())
        assertEquals(10, ds.features.rows())
        assertEquals(3, ds.labels.columns())
        assertEquals(10, ds.labels.rows())
        while (iterator.hasNext()) {
            ds = iterator.next()
            total += ds.features.rows()
        }
        assertEquals(150, total)
        assertEquals(4, ds.features.columns())
        assertEquals(10, ds.features.rows())
        assertEquals(3, ds.labels.columns())
        assertEquals(10, ds.labels.rows())
    }

}