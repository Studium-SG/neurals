package sg.studium.neurals.model

interface Model {

    /**
     * Name of the returned pairs match the trained column names (from the schema).
     * - for regression, only one
     * - for classification one for each class.
     *
     * Verifies if all explanatory variables are specified and throws exception if not.
     *
     * @param explanatory explanatory variables, should match the schema that the model was generated with
     * @return name: value pairs as the model was trained.
     */
    fun predict(explanatory: Map<String, Any>): Map<String, Double>

    /**
     * Alternative way to predict: X -> Y
     *
     * @param X X, before standardization (the model will do it)
     * @return Y
     */
    fun predict(X: DoubleArray): DoubleArray

}