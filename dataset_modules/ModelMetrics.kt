
package com.univitmil.dataset_modules

/**
 * A static data object to simulate model performance metrics for the UniViT-MIL model.
 *
 * This object provides hardcoded values for model performance, intended for use in the
 * Android application's model dashboard UI until the live model is deployed. The data
 * is based on the best results from the research paper.
 */
object ModelMetrics {

    // --- Global Performance Metrics ---
    const val GLOBAL_ACCURACY = 0.984
    const val GLOBAL_AUC = 0.997
    const val F1_SCORE = 0.982
    const val SENSITIVITY = 0.979 // Also known as Recall
    const val SPECIFICITY = 0.991
    const val INFERENCE_TIME = "120ms"

    // --- Per-Class Performance Data ---
    val classPerformances = listOf(
        ClassPerformance(
            className = "Bone",
            accuracy = 0.984,
            auc = 0.998,
            f1Score = 0.983,
            sensitivity = 0.981,
            specificity = 0.994
        ),
        ClassPerformance(
            className = "Breast",
            accuracy = 0.962,
            auc = 0.992,
            f1Score = 0.959,
            sensitivity = 0.952,
            specificity = 0.988
        ),
        ClassPerformance(
            className = "Cervical",
            accuracy = 0.975,
            auc = 0.996,
            f1Score = 0.973,
            sensitivity = 0.969,
            specificity = 0.990
        ),
        ClassPerformance(
            className = "Prostate",
            accuracy = 0.988,
            auc = 0.999,
            f1Score = 0.987,
            sensitivity = 0.985,
            specificity = 0.996
        ),
        ClassPerformance(
            className = "Endometrial",
            accuracy = 0.991,
            auc = 0.999,
            f1Score = 0.990,
            sensitivity = 0.988,
            specificity = 0.997
        )
    )
}

/**
 * Data class to hold performance metrics for a single class.
 */
data class ClassPerformance(
    val className: String,
    val accuracy: Double,
    val auc: Double,
    val f1Score: Double,
    val sensitivity: Double,
    val specificity: Double
)
