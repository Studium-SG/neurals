package sg.studium.neurals.prepare

/**
 * Maps feature:value pairs to their indices in a feature`[value`] encoded header (=categorical)
 */
class HeaderMap(
        header: Array<String>
) {

    val simples: MutableMap<String, Int> = mutableMapOf()
    val categorical: MutableMap<String, MutableMap<String, Int>> = mutableMapOf()

    init {
        val regex = "(.*)\\[(.*)\\]".toRegex()
        header.withIndex().forEach { (i, v) ->
            val matches = regex.matchEntire(v)
            if (matches == null) {
                // simple column name
                if (simples[v] != null) throw RuntimeException("Column '$v' mapped more than once")
                if (categorical[v] != null) throw RuntimeException("Column '$v' was already mapped as categorical")

                simples.put(v, i)
            } else {
                // means column in feature[value] format
                assert(matches.groupValues.size == 3)
                val feature = matches.groupValues[1]
                val value = matches.groupValues[2]

                if (simples[feature] != null) throw RuntimeException("Column '$v' mapped as single column $feature")

                var map = categorical[feature]
                if (map == null) {
                    map = mutableMapOf()
                    categorical.put(feature, map)
                }
                if (map[value] != null) throw RuntimeException("Column $v mapped as categorical multiple times")

                map.put(value, i)
            }
        }
    }

    /**
     * @return target column index if mapped or null
     */
    fun targetIndex(feature: String, value: String): Int? {
        var ret = simples[feature]
        if (ret == null) {
            val catMap = categorical[feature]
            if (catMap != null) {
                ret = catMap[value]
            }
        }
        return ret
    }

}