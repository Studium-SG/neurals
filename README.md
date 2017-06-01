# neurals
DL4J training and models for tabular data, simplified and ready to use in enterprise applications.

Design is limited to one node - no Spark. This should suffice for most.

## Classification

Input csv:
```
sepallength,sepalwidth,petallength,petalwidth,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3,1.4,0.2,Iris-setosa
...
7,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
...
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3,5.1,1.8,Iris-virginica
```

Convert to one-hot:

```kotlin
csvToOneHot({ inputstream }, outputstream )
```

```
sepallength,sepalwidth,petallength,petalwidth,class[Iris-setosa],class[Iris-versicolor],class[Iris-virginica]
5.1,3.5,1.4,0.2,1,0,0
4.9,3,1.4,0.2,1,0,0
...
7,3.2,4.7,1.4,0,1,0
6.4,3.2,4.5,1.5,0,1,0
...
6.2,3.4,5.4,2.3,0,0,1
5.9,3,5.1,1.8,0,0,1
```

See TrainClassificationInMemory.kt for training.
```kotlin
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
//          .epochTerminationConditions(MaxEpochsTerminationCondition(1000))
        .iterationTerminationConditions(MaxTimeIterationTerminationCondition(2, TimeUnit.MINUTES))
        .scoreCalculator(DataSetLossCalculator(iterator, false))
        .modelSaver(esms)
        .build()
val trainer = EarlyStoppingTrainer(esc, net, iterator)

val result = trainer.fit()
println(result)
val bestModel = result.bestModel
```

Build and save model:

```kotlin
val dl4jModel = Dl4jModel.build(iterator.getAllColumns(), iterator.getlYColumns(), normalizer, bestModel)
dl4jModel.save(FileOutputStream("/tmp/iris.onehot.dl4jmodel.zip"))
```

Predict:

```kotlin
val xy = csvToXY(resource("iris.onehot.csv"), "class")
xy.X.forEach { println(dl4jModel.predict(it).toList()) }
```

```
0: [0.9855325818061829, 0.04370792582631111, 2.0035284978803247E-4]
1: [0.9826089143753052, 0.04941225424408913, 2.1769091836176813E-4]
...
50: [0.013916244730353355, 0.8909072279930115, 0.04695480316877365]
51: [0.015126131474971771, 0.8526880741119385, 0.05702678859233856]
...
148: [0.0011371514992788434, 0.0789957270026207, 0.9534330368041992]
149: [0.002187718404456973, 0.21522334218025208, 0.8143854141235352]
```

Alternatively:

```kotlin
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
```

## Regression

Input csv:
```
vendor,MYCT,MMIN,MMAX,CACH,CHMIN,CHMAX,class
adviser,125,256,6000,256,16,128,199
amdahl,29,8000,32000,32,8,32,253
amdahl,29,8000,32000,32,8,32,253
amdahl,29,8000,32000,32,8,32,253
amdahl,29,8000,16000,32,8,16,132
amdahl,26,8000,32000,64,8,32,290
amdahl,23,16000,32000,64,16,32,381
amdahl,23,16000,32000,64,16,32,381
amdahl,23,16000,64000,64,16,32,749
amdahl,23,32000,64000,128,32,64,1238
apollo,400,1000,3000,0,1,2,23
apollo,400,512,3500,4,1,6,24
basf,60,2000,8000,65,1,8,70
basf,50,4000,16000,65,1,8,117
bti,350,64,64,0,1,4,15
bti,200,512,16000,0,4,32,64
burroughs,167,524,2000,8,4,15,23
burroughs,143,512,5000,0,7,32,29
...
```

One-hot:
```
vendor[adviser],vendor[amdahl],vendor[apollo],vendor[basf],vendor[bti],vendor[burroughs],vendor[c.r.d],vendor[cambex],vendor[cdc],vendor[dec],vendor[dg],vendor[formation],vendor[four-phase],vendor[gould],vendor[harris],vendor[honeywell],vendor[hp],vendor[ibm],vendor[ipl],vendor[magnuson],vendor[microdata],vendor[nas],vendor[ncr],vendor[nixdorf],vendor[perkin-elmer],vendor[prime],vendor[siemens],vendor[sperry],vendor[sratus],vendor[wang],MYCT,MMIN,MMAX,CACH,CHMIN,CHMAX,class
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,256,6000,256,16,128,199
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,32000,32,8,32,253
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,8000,16000,32,8,16,132
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,8000,32000,64,8,32,290
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,32000,64,16,32,381
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,32000,64,16,32,381
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,16000,64000,64,16,32,749
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,32000,64000,128,32,64,1238
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,400,1000,3000,0,1,2,23
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,400,512,3500,4,1,6,24
0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,2000,8000,65,1,8,70
0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,4000,16000,65,1,8,117
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,350,64,64,0,1,4,15
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,512,16000,0,4,32,64
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,524,2000,8,4,15,23
0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,143,512,5000,0,7,32,29
```

See TrainRegressionFromCsv.kt for training:
```kotlin
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
//      .epochTerminationConditions(MaxEpochsTerminationCondition(1000))
        .iterationTerminationConditions(MaxTimeIterationTerminationCondition(2, TimeUnit.MINUTES))
        .scoreCalculator(DataSetLossCalculator(iterator, false))
        .modelSaver(esms)
        .build()
val trainer = EarlyStoppingTrainer(esc, net, iterator)

val result = trainer.fit()
println(result)
val bestModel = result.bestModel
```

Predict:
```kotlin
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
```

## Java usage

See [Kotlin-Java-interop](https://kotlinlang.org/docs/reference/java-to-kotlin-interop.html)