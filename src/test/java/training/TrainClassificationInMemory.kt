package training

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions
import resource
import sg.studium.neurals.dataset.InMemoryDataSetIterator
import sg.studium.neurals.dataset.csvToXY
import sg.studium.neurals.model.Dl4jModel
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Speed: 14199 epochs in 2 minutes on T460p I7 nd4j-native
 * Score: 0.7069 @ epoch 14198
 */
class TrainClassificationInMemory {

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val batchSize = 10

            val iterator = InMemoryDataSetIterator(csvToXY(resource("iris.onehot.csv"), "class"), batchSize)

            val normalizer = NormalizerStandardize()
            normalizer.fit(iterator)
            iterator.preProcessor = normalizer
            NormalizerSerializer.getDefault().write(normalizer, "/tmp/iris.onehot.inmemory.normalizer")

            val netConf = NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .learningRate(0.001)
                    .iterations(1)
                    .list(
                            DenseLayer.Builder()
                                    .nIn(iterator.inputColumns())
                                    .nOut(iterator.inputColumns())
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.SIGMOID)
                                    .build(),
                            OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                                    .nIn(iterator.inputColumns())
                                    .nOut(iterator.totalOutcomes())
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.SIGMOID)
                                    .build()
                    )
                    .pretrain(false)
                    .backprop(true)
                    .build()
            val net = MultiLayerNetwork(netConf)

            val esms = InMemoryModelSaver<MultiLayerNetwork>()
            val esc = EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
//                    .epochTerminationConditions(MaxEpochsTerminationCondition(1000))
                    .iterationTerminationConditions(MaxTimeIterationTerminationCondition(2, TimeUnit.MINUTES))
                    .scoreCalculator(DataSetLossCalculator(iterator, false))
                    .modelSaver(esms)
                    .build()
            val trainer = EarlyStoppingTrainer(esc, net, iterator)

            val result = trainer.fit()
            println(result)
            val bestModel = result.bestModel

            val dl4jModel = Dl4jModel.build(iterator.getAllColumns(), iterator.getlYColumns(), normalizer, bestModel)
            dl4jModel.save(FileOutputStream("/tmp/iris.onehot.dl4jmodel.zip"))

            val xy = csvToXY(resource("iris.onehot.csv"), "class")
            xy.X.forEach { println(dl4jModel.predict(it).toList()) }

//            ModelSerializer.writeModel(bestModel, "/tmp/iris.onehot.model.inmemory.zip", true)
        }
    }
}