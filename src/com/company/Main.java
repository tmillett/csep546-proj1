package com.company;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

public class Main {

    public static void main(String[] args) {

        String validationDataPath = null;


        String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis_missing.arff";
        validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testing_tennis_missing.arff";

        //String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_subsetD.arff";
        //validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_subsetD.arff";
        ID3DecisionTree tree = new ID3DecisionTree(trainingDataPath, validationDataPath);
        tree.evaluate();


    }



}
