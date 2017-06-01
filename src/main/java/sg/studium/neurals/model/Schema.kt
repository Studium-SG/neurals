package sg.studium.neurals.model

data class Schema(
        val colsAll: List<String>,
        val colsX: List<String>,
        val colsY: List<String>
) {
    fun toColsXMap(): Map<String, Int> {
        return colsX.associate { Pair(it, colsAll.indexOf(it)) }
    }

    fun toColsYMap(): Map<String, Int> {
        return colsY.associate { Pair(it, colsAll.indexOf(it)) }
    }
}