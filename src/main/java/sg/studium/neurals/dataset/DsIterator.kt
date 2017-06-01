package sg.studium.neurals.dataset

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

interface DsIterator : DataSetIterator {

    /**
     * @return all header columns, see [XY.allColumns], might be empty if no header was provided
     */
    fun getAllColumns(): Array<String>

    /**
     * @return Y columns, see [XY.yCols], might be empty if no header was provided
     */
    fun getlYColumns(): Array<String>

}