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
    private Map<Object, List<Integer>> classInstanceCount;
    private Map<Object, Integer> attributeInstanceCount;

    public GainInfo(Attribute attr, Attribute classAttribute) {

        this.attribute = attr;
        this.classAttribute = classAttribute;

        this.classInstanceCount = new HashMap<Object, List<Integer>>();
        this.attributeInstanceCount = new HashMap<Object, Integer>();
    }

    public void addInstance(double attrValue, double classValue){
        List<Integer> classCount = this.classInstanceCount.get(new Double(attrValue));
        if (classCount == null) {
            classCount = new ArrayList<>();
            for (int k = 0; k < this.classAttribute.numValues(); k++) {
                classCount.add(new Integer(0));
            }
            this.classInstanceCount.put(new Double(attrValue), classCount);
        }

        Integer countForSpecificClass = classCount.get((int)classValue);
        classCount.set((int)classValue,countForSpecificClass + 1);

        Integer countForAttributeInstance = this.attributeInstanceCount.get(new Double(attrValue));
        if (countForAttributeInstance == null) countForAttributeInstance = new Integer(0);
        this.attributeInstanceCount.put(new Double(attrValue), countForAttributeInstance + 1);
    }

    public double gain() {
        return 0.0;
    }

    private double getEntropyForAttributeInstance(double attrValue) {
        List<Integer> classCount = this.classInstanceCount.get(new Double(attrValue));
        if (classCount == null) {
            return 0;
        }

        Integer attrCount = this.attributeInstanceCount.get(new Double((attrValue)));

        double entropy = 0.0;
        for (int i = 0; i < classCount.size(); i++) {
            int count = classCount.get(i);
            double percentage = (double)count/(double)attrCount;
            entropy -= percentage * (Math.log10(percentage) / Math.log10(2));
        }

        return entropy;
    }

    public String toString() {
        return "" + this.classInstanceCount + " " + this.attributeInstanceCount;
    }



}


