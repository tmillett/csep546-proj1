package com.company;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by tmillett on 4/10/17.
 */
public class GainInfo {

    private Attribute attribute;
    private Attribute classAttribute;
    private Double attributeInstance;
    private Map<Double, List<Integer>> classInstanceCount;
    private Map<Double, Integer> attributeInstanceCount;
    private int totalCount;

    public GainInfo(Attribute attr, Attribute classAttribute) {

        this.attribute = attr;
        this.classAttribute = classAttribute;

        this.classInstanceCount = new HashMap<>();
        this.attributeInstanceCount = new HashMap<>();
        this.totalCount = 0;
    }

    public void addInstance(double attrValue, double classValue){
        List<Integer> classCount = this.classInstanceCount.get(attrValue);
        if (classCount == null) {
            classCount = new ArrayList<>();
            for (int k = 0; k < this.classAttribute.numValues(); k++) {
                classCount.add(0);
            }
            this.classInstanceCount.put(attrValue, classCount);
        }

        Integer countForSpecificClass = classCount.get((int)classValue);
        classCount.set((int)classValue,countForSpecificClass + 1);

        Integer countForAttributeInstance = this.attributeInstanceCount.get(attrValue);
        if (countForAttributeInstance == null) countForAttributeInstance = 0;
        this.attributeInstanceCount.put(attrValue, countForAttributeInstance + 1);

        this.totalCount++;
    }

    public double gain(double tableEntropy) {

        double gain = 0.0;
        for (Double attrValue : this.classInstanceCount.keySet()) {
            Integer attrCount = this.attributeInstanceCount.get(attrValue);

            double prefix = (double)attrCount / (double)this.totalCount;
            gain += prefix * this.getEntropyForAttributeInstance(attrValue);
        }

        return tableEntropy - gain;
    }

    private double getEntropyForAttributeInstance(double attrValue) {
        List<Integer> classCount = this.classInstanceCount.get(attrValue);
        if (classCount == null) {
            return 0;
        }

        Integer attrCount = this.attributeInstanceCount.get(attrValue);

        double entropy = 0.0;
        for (int i = 0; i < classCount.size(); i++) {
            int count = classCount.get(i);
            if (count != attrCount && count > 0) {
                double percentage = (double) count / (double) attrCount;
                double loggedPercent = Math.log(percentage);
                double loggedBase2 = Math.log(2);
                double convertedPercentage = loggedPercent / loggedBase2;

                entropy -= percentage * convertedPercentage;
            }
        }

        return entropy;
    }

    public String toString() {
        return "" + this.classInstanceCount + " " + this.attributeInstanceCount;
    }

}


