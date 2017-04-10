package com.company;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import java.util.*;

public class Main {

    public static void main(String[] args) {

        try {
            DataSource source = new DataSource("/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis.arff");


            Instances data = source.getDataSet();

            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);


            System.out.println(data.attribute(0).numValues());


            findBestAttribute(data);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void growTree(Instances data, DataSource source) {

        growTree(data, source);
    }

    public static Attribute findBestAttribute(Instances data) {
        // C_i = percent of ith class value to total number of all classes
        // Calculate the entropy: E(S) = Sum(-p(C_i)log_2(C_i))
        int totalNumOfRows = data.numInstances();
        AttributeStats classStats = data.attributeStats(data.classIndex());

        double entropy = 0.0;
        for (int i = 0; i < classStats.nominalCounts.length; i++) {
            int count = classStats.nominalCounts[i];
            double percentage = (double)count/(double)totalNumOfRows;
            entropy -= percentage * (Math.log10(percentage) / Math.log10(2));
        }

        System.out.println(entropy);

        /*
        Calculate the gain for each attribute
        Data Structure is following HashMap
        Key is attribute class
        Value is another HashMap with Key being specific enumerations of the attribute,
        Value being array of size [numClassValues] and counts of each type
        eg {{M,F} ->{ M-> [2,3,1], F -> [4, 0, 1]}}
        */

        Map<Attribute, GainInfo> instanceRowsByClass = new HashMap<>(data.size());
        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.instance(i);
            double classValue = instance.classValue();
            int numClassValues = instance.classAttribute().numValues();
            for (int j = 0; j < instance.numAttributes(); j++) {
                Attribute attr = instance.attribute(j);
                double attrValue = instance.value(j);
                GainInfo classCountPerAttrType = instanceRowsByClass.get(attr);
                if (classCountPerAttrType == null) {
                    classCountPerAttrType = new GainInfo(attr,instance.classAttribute());
                    instanceRowsByClass.put(attr, classCountPerAttrType);
                }

                classCountPerAttrType.addInstance(attrValue, classValue);
            }
        }

        System.out.println(instanceRowsByClass);
        Attribute attribueWithHighestGain = null;
        double highestGain = Double.MIN_VALUE;


        return null;



    }


}
