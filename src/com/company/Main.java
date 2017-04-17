package com.company;

public class Main {

    public static void main(String[] args) {

        String validationDataPath = null;

        //String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis.arff";
        //validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testing_tennis.arff";


        //String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis_missing.arff";
        //validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testing_tennis_missing.arff";

        String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_subsetD.arff";
        validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testingD.arff";
        ID3DecisionTree tree = new ID3DecisionTree(trainingDataPath, validationDataPath);
        tree.evaluate();


    }



}
