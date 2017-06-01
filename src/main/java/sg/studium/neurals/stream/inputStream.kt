package sg.studium.neurals.stream

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream
import java.io.FileInputStream
import java.io.InputStream
import java.util.zip.GZIPInputStream

/**
 * Creates an [InputStream] from a file name with transparent decompression.
 *
 * Supported suffixes: .gz, .bz2, .xz
 */
fun inputStream(fileName: String): InputStream {
    val fis = FileInputStream(fileName)
    return when {
        fileName.endsWith(".gz") -> GZIPInputStream(fis)
        fileName.endsWith(".bz2") -> BZip2CompressorInputStream(fis)
        fileName.endsWith(".xz") -> XZCompressorInputStream(fis)
        else -> fis
    }
}
