package org.crislmfroes.moneynet

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.util.concurrent.Executors

private const val REQUEST_CODE_PERMISSIONS = 10

// This is an array of all the permission specified in the manifest.
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

class MainActivity : AppCompatActivity() {

    private lateinit var textView: TextView

    private lateinit var interpreter: Interpreter

    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Add this at the end of onCreate function

        viewFinder = findViewById(R.id.view_finder)

        textView = findViewById(R.id.text_view)

        val model = FileUtil.loadMappedFile(this, "money_classifier.tflite")

        labels = FileUtil.loadLabels(this, "labels.txt")

        interpreter = Interpreter(model)

        // Request camera permissions
        if (allPermissionsGranted()) {
            viewFinder.post { startCamera() }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Every time the provided texture view changes, recompute layout
        viewFinder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }
    }

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var viewFinder: TextureView

    private fun startCamera() {
        // Create configuration object for the viewfinder use case
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(640, 480))
        }.build()


        // Build the viewfinder use case
        val preview = Preview(previewConfig)

        // Every time the viewfinder is updated, recompute layout
        preview.setOnPreviewOutputUpdateListener {

            // To update the SurfaceTexture, we have to remove it and re-add it
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)

            viewFinder.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        // Add this before CameraX.bindToLifecycle

        // Setup image analysis pipeline that computes average pixel luminance
        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            // In our analysis, we care more about the latest image than
            // analyzing *every* image
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
        }.build()

        // Build the image analysis use case and instantiate our analyzer
        val moneyAnalyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, MoneyAnalyzer())
        }

        // Bind use cases to lifecycle
        // If Android Studio complains about "this" being not a LifecycleOwner
        // try rebuilding the project or updating the appcompat dependency to
        // version 1.1.0 or higher.
        CameraX.bindToLifecycle(this, preview, moneyAnalyzerUseCase)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // Compute the center of the view finder
        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        // Correct preview output to account for display rotation
        val rotationDegrees = when(viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix)
    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                viewFinder.post { startCamera() }
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private inner class MoneyAnalyzer : ImageAnalysis.Analyzer {

        var lastAnalyzed = System.currentTimeMillis()

        override fun analyze(imageProxy: ImageProxy?, degrees: Int) {
            if (System.currentTimeMillis() - lastAnalyzed > 1000/60.0) {
                val bitmap = Bitmap.createScaledBitmap(viewFinder.bitmap, 224, 224, false)
                val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
                val output = Array(1) { FloatArray(labels.size) }
                bitmap?.let {
                    for (x in 0 until it.width) {
                        for (y in 0 until it.height) {
                            val pixel = it.getPixel(x, y)
                            input[0][x][y][0] = ((pixel shr 16 and 0xff)/127.5f) - 1
                            input[0][x][y][1] = ((pixel shr 8 and 0xff)/127.5f) - 1
                            input[0][x][y][2] = ((pixel and 0xff)/127.5f) - 1
                        }
                    }
                    interpreter.run(input, output)
                    var maxProb = 0f
                    var maxIndex = 0
                    for (i in output[0].indices) {
                        if (output[0][i] > maxProb) {
                            maxProb = output[0][i]
                            maxIndex = i
                        }
                    }
                    val className = labels[maxIndex].replace('_', ' ').capitalize()
                    var message = "%s: %.2f%%".format(className, maxProb*100)
                    if (labels[maxIndex] == "background") {
                        message = "Nenhum dinheiro detectado."
                    } else if (maxProb < 0.4) {
                        message = "Incerteza ao identificar dinheiro. Coloque a nota extendida em uma superfícia um pouco à frente do celular."
                    }
                    runOnUiThread {
                        textView.text = message
                        Log.i("MoneyAnalyzer", message)
                    }
                }
                lastAnalyzed = System.currentTimeMillis()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }
}