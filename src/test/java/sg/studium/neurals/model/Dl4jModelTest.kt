package sg.studium.neurals.model

import org.apache.commons.io.IOUtils
import org.datavec.api.transform.schema.Schema
import org.deeplearning4j.util.ModelSerializer
import org.junit.Test
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer
import resource
import sg.studium.neurals.dataset.csvToXY
import java.io.FileOutputStream
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class Dl4jModelTest {

    @Test
    fun buildDl4jModel() {
        val schema = Schema.fromYaml(IOUtils.toString(resource("cpu.onehot.schema.yaml")))
        val normalizer = NormalizerSerializer.getDefault().restore<NormalizerStandardize>(resource("cpu.onehot.inmemory.normalizer"))
        val model = ModelSerializer.restoreMultiLayerNetwork(resource("cpu.onehot.model.inmemory.zip"))

        val dl4jModel = Dl4jModel.build(schema.columnNames, listOf("class"), normalizer, model)
        val fos = FileOutputStream("/tmp/cpu.onehot.inmemory.dl4jmodel.zip")
        dl4jModel.save(fos)
        fos.close()
    }

    @Test
    fun regressionPredictMap() {
        val dl4jModel = Dl4jModel.load(resource("cpu.onehot.inmemory.dl4jmodel.zip"))
        val output = dl4jModel.predict(mapOf(
                "vendor" to "adviser",
                "MYCT" to 125,
                "MMIN" to 256,
                "MMAX" to 6000,
                "CACH" to 256,
                "CHMIN" to 16,
                "CHMAX" to 128
        ))
        assertEquals(1, output.size)
        val value = output["class"]
        assertNotNull(value)
        assertTrue(Math.abs(value!! - 199.0) < 0.5)
    }

    @Test
    fun regressionPredictArray() {
        val dl4jModel = Dl4jModel.load(resource("cpu.onehot.inmemory.dl4jmodel.zip"))
        val output = dl4jModel.predict(doubleArrayOf(
                1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                125.0, 256.0, 6000.0, 256.0, 16.0,
                128.0))
        assertEquals(1, output.size)
        assertTrue(Math.abs(output[0] - 199.0) < 0.5)
    }

    @Test
    fun classificationPredictMap() {
        val dl4jModel = Dl4jModel.load(resource("iris.onehot.dl4jmodel.zip"))
        val output = dl4jModel.predict(mapOf(
                "sepallength" to 5.1,
                "sepalwidth" to 3.5,
                "petallength" to 1.4,
                "petalwidth" to 0.2
        ))
        assertEquals(3, output.size)
        assertTrue(output["class[Iris-setosa]"]!! > 0.98)
        assertTrue(output["class[Iris-versicolor]"]!! < 0.05)
        assertTrue(output["class[Iris-virginica]"]!! < 0.02)
    }

//    @Test
//    fun classificationPredict() {
//        val dl4jModel = Dl4jModel.load(resource("iris.onehot.dl4jmodel.zip"))
//        val xy = csvToXY(resource("iris.onehot.csv"), "class")
//        xy.X.withIndex().forEach {
//            println("${it.index}: ${dl4jModel.predict(it.value).toList()}")
//        }
//    }

    @Test
    fun classificationPredictArray() {
        val dl4jModel = Dl4jModel.load(resource("iris.onehot.dl4jmodel.zip"))
        val output = dl4jModel.predict(doubleArrayOf(5.1, 3.5, 1.4, 0.2))
        assertEquals(3, output.size)
        assertTrue(output[0] > 0.98)
        assertTrue(output[1] < 0.05)
        assertTrue(output[2] < 0.02)
    }
}