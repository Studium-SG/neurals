package sg.studium.neurals.model

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import org.apache.commons.io.IOUtils
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer
import org.nd4j.linalg.factory.Nd4j
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

class Dl4jModel(
        private val colsX: List<String>,
        private val colsY: List<String>,
        private val normalizer: NormalizerStandardize?,
        private val net: MultiLayerNetwork
) : Model {

    private val colsXMap: Map<String, Int> by lazy {
        colsX.withIndex().associate { (idx, v) -> Pair(v, idx) }
    }

    override fun predict(explanatory: Map<String, Any>): Map<String, Double> {
        val features = DoubleArray(colsX.size, { 0.0 })
        explanatory.forEach { (s, v) ->
            if (v is String) {
                // categorical, "feature[value]" case, set to 1.0
                features[colsXMap["$s[$v]"]!!] = 1.0
            } else if (v is Number) {
                // numeric case
                features[colsXMap[s]!!] = v.toDouble()
            } else {
                throw RuntimeException("Should be String or Number, got ${v.javaClass}")
            }
        }
        val output = predict(features)
        return output.withIndex().associate { (idx, value) -> Pair(colsY[idx], value) }
    }

    override fun predict(X: DoubleArray): DoubleArray {
        val features = Nd4j.create(X)
        normalizer?.transform(features)
        val output = net.output(features)
        assert(output.rows() == 1)
        assert(output.columns() == colsY.size)
        return DoubleArray(output.columns(), { output.getDouble(it) })
    }

    /**
     * Saves the model into a zip file.
     *
     * Recommended suffix: ".dl4jmodel.zip"
     */
    fun save(outputStream: OutputStream) {
        val mapper = ObjectMapper().registerModule(KotlinModule())
        val zout = ZipOutputStream(outputStream)

        zout.putNextEntry(ZipEntry("colsX.json"))
        zout.write(mapper.writeValueAsBytes(colsX))
        zout.closeEntry()

        zout.putNextEntry(ZipEntry("colsY.json"))
        zout.write(mapper.writeValueAsBytes(colsY))
        zout.closeEntry()

        if (normalizer != null) {
            val baosNormalizer = ByteArrayOutputStream()
            NormalizerSerializer.getDefault().write(normalizer, baosNormalizer)

            zout.putNextEntry(ZipEntry("normalizer.data"))
            zout.write(baosNormalizer.toByteArray())
            zout.closeEntry()
        }

        val modelBaos = ByteArrayOutputStream()
        ModelSerializer.writeModel(net, modelBaos, true)

        zout.putNextEntry(ZipEntry("model.dl4j.zip"))
        zout.write(modelBaos.toByteArray())
        zout.closeEntry()

        zout.close()
    }

    companion object {

        /**
         * @param allColumnNames list of all column names, X and Y
         * @param colsY list of Y column names
         * @param normalizer optional normalizer that was used to train the model
         * @param net the trained network
         * @return created model
         */
        fun build(allColumnNames: List<String>,
                  colsY: List<String>,
                  normalizer: NormalizerStandardize?,
                  net: MultiLayerNetwork): Dl4jModel {
            val colsX = allColumnNames.filter { scIt -> !colsY.any { scIt.startsWith("$it[") || scIt == it } }
            return Dl4jModel(colsX, colsY, normalizer, net)
        }

        /**
         * @param allColumnNames array of all column names, X and Y
         * @param colsY array of Y column names
         * @param normalizer optional normalizer that was used to train the model
         * @param net the trained network
         * @return created model
         */
        fun build(allColumnNames: Array<String>,
                  colsY: Array<String>,
                  normalizer: NormalizerStandardize?,
                  net: MultiLayerNetwork): Dl4jModel {
            return build(allColumnNames.toList(), colsY.toList(), normalizer, net)
        }

        /**
         * Loads the model from a zip file.
         *
         * Recommended suffix: ".dl4jmodel.zip"
         */
        fun load(inputStream: InputStream): Dl4jModel {
            val mapper = ObjectMapper().registerModule(KotlinModule())

            var colsX: List<String>? = null
            var colsY: List<String>? = null
            var normalizer: NormalizerStandardize? = null
            var net: MultiLayerNetwork? = null

            val zin = ZipInputStream(inputStream)
            var entry = zin.nextEntry
            while (entry != null) {
                val bar = IOUtils.toByteArray(zin)
                @Suppress("UNCHECKED_CAST")
                when {
                    entry.name == "colsX.json" -> colsX = mapper.readValue(bar, List::class.java) as List<String>
                    entry.name == "colsY.json" -> colsY = mapper.readValue(bar, List::class.java) as List<String>
                    entry.name == "normalizer.data" -> normalizer = NormalizerSerializer.getDefault().restore(ByteArrayInputStream(bar))
                    entry.name == "model.dl4j.zip" -> net = ModelSerializer.restoreMultiLayerNetwork(ByteArrayInputStream(bar), true)
                    else -> throw RuntimeException("Unknown entry: ${entry.name}")
                }
                entry = zin.nextEntry
            }

            if (colsX == null) throw RuntimeException("Zip file does not have 'colsX.json'")
            if (colsY == null) throw RuntimeException("Zip file does not have 'colsY.json'")
            if (net == null) throw RuntimeException("Zip file does not have 'model.dl4j.zip'")

            return Dl4jModel(
                    colsX,
                    colsY,
                    normalizer,
                    net)
        }
    }
}