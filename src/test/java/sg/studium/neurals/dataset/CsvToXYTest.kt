package sg.studium.neurals.dataset

import org.junit.Test
import resource
import kotlin.test.assertEquals

class CsvToXYTest {

    @Test
    fun simpleLabelWithIndices() {
        val xy =  csvToXY(resource("cpu.onehot.data.csv"), 36, 36)
        assertEquals(xy.X.size, 209)
        assertEquals(xy.X.first().size, 36)
        assertEquals(xy.Y.size, 209)
        assertEquals(xy.Y.first().size, 1)
        assertEquals(xy.Y.first()[0], 199.0)
    }

    @Test
    fun simpleLabelWithName() {
        val xy =  csvToXY(resource("cpu.onehot.data.csv"), "class")
        assertEquals(xy.X.size, 209)
        assertEquals(xy.X.first().size, 36)
        assertEquals(xy.Y.size, 209)
        assertEquals(xy.Y.first().size, 1)
        assertEquals(xy.Y.first()[0], 199.0)
    }

    @Test
    fun multiclassLabelWithIndices() {
        val xy =  csvToXY(resource("iris.onehot.csv"), 4, 6)
        assertEquals(xy.X.size, 150)
        assertEquals(xy.X.first().size, 4)
        assertEquals(xy.Y.size, 150)
        assertEquals(xy.Y.first().size, 3)
        assertEquals(xy.X.first()[0], 5.1)
        assertEquals(xy.X.first()[1], 3.5)
        assertEquals(xy.X.first()[2], 1.4)
        assertEquals(xy.X.first()[3], 0.2)
        assertEquals(xy.Y.first()[0], 1.0)
        assertEquals(xy.Y.first()[1], 0.0)
        assertEquals(xy.Y.first()[2], 0.0)
    }

    @Test
    fun multiclassLabelWithName() {
        val xy =  csvToXY(resource("iris.onehot.csv"), "class")
        assertEquals(xy.X.size, 150)
        assertEquals(xy.X.first().size, 4)
        assertEquals(xy.Y.size, 150)
        assertEquals(xy.Y.first().size, 3)
        assertEquals(xy.X.first()[0], 5.1)
        assertEquals(xy.X.first()[1], 3.5)
        assertEquals(xy.X.first()[2], 1.4)
        assertEquals(xy.X.first()[3], 0.2)
        assertEquals(xy.Y.first()[0], 1.0)
        assertEquals(xy.Y.first()[1], 0.0)
        assertEquals(xy.Y.first()[2], 0.0)
    }

    @Test
    fun missingValues() {
        val xy =  csvToXY(resource("cpu.with.vendor.missing.onehot.csv"), "class")
        assertEquals(xy.X.size, 11)
        assertEquals(xy.X.first().size, 36)
        assertEquals(xy.Y.size, 11)
        assertEquals(xy.Y.first().size, 1)
    }
}