package com.company;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

public class Main {

    public static void main(String[] args) {

        try {

            Instances data = getInstances();
            ID3TreeNode tree = new ID3TreeNode(null);
            tree.train(data);
            System.out.println(data.attribute(data.classIndex()));
            tree.print();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static Instances getInstances() throws Exception {
        DataSource source = new DataSource("/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis.arff");


        Instances data = source.getDataSet();

        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

}
