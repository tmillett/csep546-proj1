package com.company;

/**
 * Created by tmillett on 4/17/17.
 */
public class CriticalValuesTable {

    private static double[][] criticalValues;

    public CriticalValuesTable () {
        this.criticalValues = new double[][]{
                { 3.84, 6.64},
                { 5.99, 9.21},
                { 7.82, 11.34},
                { 9.49, 13.28},
                { 11.07, 15.09},
                { 12.59, 16.81},
                { 14.07, 18.48},
                { 15.51, 20.09},
                { 16.92, 21.67},
                { 18.31, 23.21},
                { 19.68, 24.72},
                { 21.03, 26.22},
                { 22.36, 27.69},
                { 23.68, 29.14},
                { 25.0, 30.58},
                { 26.3, 32.0},
                { 27.59, 33.41},
                { 28.87, 34.8},
                { 30.14, 37.57},
                { 31.41, 37.57},
                { 32.67, 38.93},
                { 33.92, 40.29},
                { 35.17, 41.64},
                { 36.42, 42.98},
                { 37.65, 44.31},
                { 38.88, 45.64},
                { 40.11, 46.96},
                { 41.34, 48.28},
                { 42.56, 49.59},
                { 43.77, 50.89}

        };
    }

    // confidence: use 0 for 95% and 1 for 99%
    public double getChiSquaredValue(int numAttrValues, int confidence) {
        if (numAttrValues > 30 || numAttrValues < 0 || confidence > 1  || confidence < 0) {
            return Double.NaN;
        }

        return this.criticalValues[numAttrValues-1][confidence];
    }
}
