package com.company;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class Main {

    public static void main(String[] args) {

        try {

            Instances data = getInstances();
            Map<Double, Map<Attribute, Double>> highestAttribueValueCounts = findHighestAttributeValueCounts(data);
            replaceMissingValues(data, highestAttribueValueCounts);
            ID3TreeNode tree = new ID3TreeNode(null);
            tree.train(data);
            System.out.println(data.attribute(data.classIndex()));
            tree.print();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static void replaceMissingValues(Instances data, Map<Double, Map<Attribute, Double>> highestAttributeValueCounts) {
        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.get(i);
            Double classValue = instance.classValue();
            for (int j = 0; j < instance.numValues(); j++) {
                if (j == instance.classIndex()) continue;

                Attribute attr = instance.attribute(j);
                Double attrValue = instance.value(j);

                if (instance.isMissing(j) || attr.value(attrValue.intValue()).equals("NULL")) {
                    Map<Attribute, Double> countsForSpecificAttribute = highestAttributeValueCounts.get(classValue);
                    Double replacementAttrValue = countsForSpecificAttribute.get(attr);
                    instance.setValue(attr,replacementAttrValue);
                }
            }
        }
    }

    private static Instances getInstances() throws Exception {
        DataSource source = new DataSource("/Users/tmillett/_dev/git/uw/machine_learning/_proj1/training_tennis_missing.arff");


        Instances data = source.getDataSet();

        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private static Map<Double, Map<Attribute, Double>> findHighestAttributeValueCounts(Instances data) {
        // Loop through all instances. Find out the most common value for each attribute depending on the class value

        // First get the counts of each variable depending on the class value
        // {classValue1 -> {attribute1 -> {attributeValue1 -> count, attributeValue2 -> count}, attribute2 -> {attributeValue3 -> count, attributeValue4 -> count}},
        //  classValue2 -> {attribute1 -> {attributeValue1 -> count, attributeValue2 -> count}, attribute2 -> {attributeValue3 -> count, attributeValue4 -> count}}

        Map<Double, Map<Attribute, Map<Double, Integer>>> countsForEachAttributePerClassValue = new HashMap<>();

        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.get(i);
            Double classValue = instance.classValue();
            Map<Attribute, Map<Double, Integer>> countsForEachAttribute = countsForEachAttributePerClassValue.get(classValue);
            if (countsForEachAttribute == null) {
                countsForEachAttribute = new HashMap<>();
                countsForEachAttributePerClassValue.put(classValue,countsForEachAttribute);
            }
            for (int j = 0; j < instance.numValues(); j++) {
                if (j == instance.classIndex()) continue;
                if (instance.isMissing(j)) {
                    continue;
                }
                Attribute attr = instance.attribute(j);
                Double attrValue = instance.value(j);
                if (attr.value(attrValue.intValue()).equals("NULL")) {
                    continue;
                }
                Map<Double, Integer> countsForSpecificAttribute = countsForEachAttribute.get(attr);
                if (countsForSpecificAttribute == null) {
                    countsForSpecificAttribute = new HashMap<>();
                    countsForEachAttribute.put(attr,countsForSpecificAttribute);
                }

                Integer countForSpecificAttributeValue = countsForSpecificAttribute.get(attrValue);
                if (countForSpecificAttributeValue == null) {
                    countForSpecificAttributeValue = 0;
                }
                countsForSpecificAttribute.put(attrValue, ++countForSpecificAttributeValue);
            }
        }

        // {classValue1 -> {attribute1 -> attributeValue1, attribute2 -> attributeValue2},
        //  classValue2 -> {attribute1 -> attributeValue1, attribute2 -> attributeValue2}}
        Map<Double, Map<Attribute, Double>> highestAttributeValueCounts = new HashMap<>();

        for (Double classValue: countsForEachAttributePerClassValue.keySet()) {
            Map<Attribute, Map<Double, Integer>> countsForEachAttribute = countsForEachAttributePerClassValue.get(classValue);
            for (Attribute attr: countsForEachAttribute.keySet()) {
                Map<Double, Integer> countsForSpecificAttribute = countsForEachAttribute.get(attr);
                Integer highestCount = Integer.MIN_VALUE;
                Double highestAttrCountValue = null;
                for (Double attrValue: countsForSpecificAttribute.keySet()) {
                    Integer currentCount = countsForSpecificAttribute.get(attrValue);
                    if (highestCount < currentCount) {
                        highestCount = currentCount;
                        highestAttrCountValue = attrValue;
                    }
                }

                Map<Attribute, Double> highestAttribueCountForSpecificAttribute = highestAttributeValueCounts.get(classValue);
                if (highestAttribueCountForSpecificAttribute == null) {
                    highestAttribueCountForSpecificAttribute = new HashMap<>();
                    highestAttributeValueCounts.put(classValue, highestAttribueCountForSpecificAttribute);
                }
                highestAttribueCountForSpecificAttribute.put(attr, highestAttrCountValue);
            }
        }

        return highestAttributeValueCounts;
    }

}
