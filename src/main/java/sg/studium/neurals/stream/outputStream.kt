package sg.studium.neurals.stream

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream
import org.apache.commons.compress.compressors.xz.XZCompressorOutputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.GZIPOutputStream

/**
 * Creates an [InputStream] from a file name with transparent decompression.
 *
 * Supported suffixes: .gz, .bz2, .xz
 */
fun outputStream(fileName: String): OutputStream {
    val fos = FileOutputStream(fileName)
    return when {
        fileName.endsWith(".gz") -> GZIPOutputStream(fos)
        fileName.endsWith(".bz2") -> BZip2CompressorOutputStream(fos)
        fileName.endsWith(".xz") -> XZCompressorOutputStream(fos)
        else -> fos
    }
}
