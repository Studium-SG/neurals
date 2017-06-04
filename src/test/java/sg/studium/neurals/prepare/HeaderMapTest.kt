package sg.studium.neurals.prepare

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertNull

class HeaderMapTest {

    @Test
    fun legals() {
        val mapper = HeaderMap(arrayOf("f1[v1]", "f1[v2]", "f2", "f3"))

        assertEquals(0, mapper.targetIndex("f1", "v1"))
        assertEquals(1, mapper.targetIndex("f1", "v2"))
        assertNull(mapper.targetIndex("f1", "nonexistent"))
        assertEquals(2, mapper.targetIndex("f2", "Anything"))
        assertEquals(3, mapper.targetIndex("f3", "5555.5"))
    }

    @Test
    fun illegal1() {
        assertFails {
            val mapper = HeaderMap(arrayOf("f1[v1]", "f1[v1]"))
        }
    }

    @Test
    fun illegal2() {
        assertFails {
            val mapper = HeaderMap(arrayOf("f1[v1]", "f1"))
        }
    }

    @Test
    fun illegal3() {
        assertFails {
            val mapper = HeaderMap(arrayOf("f1", "f1[v1]"))
        }
    }

    @Test
    fun illegal4() {
        assertFails {
            val mapper = HeaderMap(arrayOf("f1", "f1"))
        }
    }
}