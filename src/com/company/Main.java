package com.company;

public class Main {

    public static void main(String[] args) {

        String trainingDataPath;
        String validationDataPath = null;
        int confidenceLevel = 0;

        if (args.length > 0) {
            trainingDataPath = args[0];
            if (args.length > 1) {
                validationDataPath = args[1];
                if (args.length > 2) {
                    confidenceLevel = Integer.parseInt(args[2]);
                }
            }

        } else {

            // Defaults

            //String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis.arff";
            //validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testing_tennis.arff";


            //String trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis_missing.arff";
            //validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testing_tennis_missing.arff";

            trainingDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_subsetD.arff";
            validationDataPath = "/Users/tmillett/_dev/git/uw/machine_learning/_proj1/testingD.arff";

            confidenceLevel = 2;
        }


        ID3DecisionTree tree = new ID3DecisionTree(trainingDataPath, validationDataPath);
        tree.evaluate(confidenceLevel);
    }



}
