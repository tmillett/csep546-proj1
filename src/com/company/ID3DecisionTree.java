package com.company;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by tmillett on 4/13/17.
 */
public class ID3DecisionTree {

    private Instances trainingData;
    private Instances validationData;

    public ID3DecisionTree(String trainingDataPath, String validationDataPath) {

        if (validationDataPath != null) {
            this.trainingData = getInstances(trainingDataPath);
            this.validationData = getInstances(validationDataPath);
        } else {

            this.trainingData = getInstances(trainingDataPath);

            this.validationData = new Instances(this.trainingData, this.trainingData.size());
            for (int i = 0; i < this.trainingData.size(); i = i + 10) {
                Instance instance = this.trainingData.remove(i);
                validationData.add(instance);
            }
        }
    }

    public void evaluate(Integer confidenceLevel) {

        ID3TreeNode tree = new ID3TreeNode(null, confidenceLevel);
        tree.train(this.trainingData);
        System.out.println(this.trainingData.attribute(this.trainingData.classIndex()));
        tree.print();

       int numMatches = 0;
        for (int i = 0; i < this.validationData.size(); i++) {
            Instance instance = this.validationData.get(i);
            Double expectedClassValue = tree.evaluateInstance(instance);
            Double actualClassValue = instance.classValue();
            if (expectedClassValue.equals(actualClassValue)) {
                numMatches++;
            } else {
                //System.out.println(instance);
            }
        }
        double percent = (double)numMatches / (double)this.validationData.size();
        System.out.println("Accuracy: " + percent);
    }

    private Instances getInstances(String path) {

        Instances data = null;
        try {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);


            data = source.getDataSet();

            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }
}
