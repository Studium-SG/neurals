package sg.studium.neurals.dataset

class XY(
        val X: List<DoubleArray>,
        val Y: List<DoubleArray>,
        /**
         * All header column names, e.g. ["sepallength", "sepalwidth", "petallength", "petalwidth", "class[Iris-setosa]", "class[Iris-versicolor]", "class[Iris-virginica]"].
         * Might be empty if no header was provided.
         */
        val allColumns: Array<String>,
        /**
         * Label column names, e.g. ["class[Iris-setosa]", "class[Iris-versicolor]", "class[Iris-virginica]"]
         * Might be empty if no header was provided.
         */
        val yCols: Array<String>
)