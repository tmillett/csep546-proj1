package com.company;

import org.w3c.dom.Attr;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by tmillett on 4/10/17.
 */
public class ID3TreeNode {

    private ID3TreeNode parent;
    private Attribute attribute;
    private Map<Double, ID3TreeNode> children;

    public ID3TreeNode(ID3TreeNode parent) {
        this.parent = parent;
        this.children = new HashMap<>();
    }

    public void setAttribute(Attribute attribute) {
        this.attribute = attribute;
    }

    public void setChildForAttributeValue(double attributeValue, ID3TreeNode node) {
        this.children.put(attributeValue, node);
    }

    public Attribute getAttribute() {
        return this.attribute;
    }

    public Map<Double, ID3TreeNode> getChildren() {
        return this.children;
    }

    private Attribute findBestAttribute(Instances data) {
        // C_i = percent of ith class value to total number of all classes
        // Calculate the entropy: E(S) = Sum(-p(C_i)log_2(C_i))
        double entropy = getTableEntropy(data);

        // Create structure to aid in calculating gain
        Map<Attribute, GainInfo> instanceRowsByClass = getAttributeGainInfoMap(data);

        Attribute attributeWithHighestGain = calculateGain(entropy, instanceRowsByClass);

        return attributeWithHighestGain;
    }

    private double getTableEntropy(Instances data) {
        int totalNumOfRows = data.numInstances();
        AttributeStats classStats = data.attributeStats(data.classIndex());

        double entropy = 0.0;
        for (int i = 0; i < classStats.nominalCounts.length; i++) {
            int count = classStats.nominalCounts[i];
            if (count != totalNumOfRows && count > 0) {
                double percentage = (double) count / (double) totalNumOfRows;
                entropy -= percentage * (Math.log10(percentage) / Math.log10(2));
            }
        }
        return entropy;
    }

    private Attribute calculateGain(double entropy, Map<Attribute, GainInfo> instanceRowsByClass) {
        Attribute attribueWithHighestGain = null;
        double highestGain = Double.MIN_VALUE;
        for (Attribute attr : instanceRowsByClass.keySet()) {
            double currentGain = instanceRowsByClass.get(attr).gain(entropy);
            if (currentGain > highestGain) {
                attribueWithHighestGain = attr;
                highestGain = currentGain;
            }

        }
        return attribueWithHighestGain;
    }

    private Map<Attribute, GainInfo> getAttributeGainInfoMap(Instances data) {
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
            for (int j = 0; j < instance.numValues(); j++) {

                if (j == instance.classIndex()) continue;

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
        return instanceRowsByClass;
    }

    public void train(Instances data) {

        Attribute root = this.findBestAttribute(data);
        this.setAttribute(root);
        int attributeIndex = -1;
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr == root) {
                attributeIndex = i;
                break;
            }
        }
        Map<Double, Instances> subInstancesMap = new HashMap<>();
        Map<Double, Double> subInstancesClassValueMap = new HashMap<>();
        Map<Double, Boolean> subInstancesShouldGrowMap = new HashMap<>();

        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.instance(i);
            double attrValue = instance.value(root);
            double classValue = instance.classValue();

            Instances subInstances = subInstancesMap.get(attrValue);
            if (subInstances == null) {
                subInstances = new Instances(data, data.size());
                subInstancesMap.put(attrValue, subInstances);
            }
            subInstances.add(instance);

            Double existingClassValue = subInstancesClassValueMap.get(attrValue);
            if (existingClassValue == null) {
                existingClassValue = classValue;
                subInstancesClassValueMap.put(attrValue, existingClassValue);
            }

            if (existingClassValue != classValue) {
                subInstancesShouldGrowMap.put(attrValue, true);
            }
        }

        for (Double attrValue : subInstancesMap.keySet()) {
            Instances subInstances = subInstancesMap.get(attrValue);
            Boolean shouldGrowTree = subInstancesShouldGrowMap.get(attrValue);
            if (shouldGrowTree != null && shouldGrowTree == true) {
                subInstances.deleteAttributeAt(attributeIndex);
                ID3TreeNode childNode = new ID3TreeNode(this);
                this.setChildForAttributeValue(attrValue, childNode);
                childNode.train(subInstances);
            } else {
                Double existingClassValue = subInstancesClassValueMap.get(attrValue);

                ID3TreeLeaf leafNode = new ID3TreeLeaf(this);

                leafNode.setClassValueForAttributeValue(attrValue, existingClassValue);
                this.setChildForAttributeValue(attrValue, leafNode);
            }
        }
    }

    public void print() {
        print("", true);
    }

    private void print(String prefix, boolean isTail) {
        printThis(prefix, isTail);
        List<ID3TreeNode> children = new ArrayList<>();
        for (Double attrValue : this.children.keySet()) {
            children.add(this.children.get(attrValue));
        }
        for (int i = 0; i < children.size() - 1; i++) {
            children.get(i).print(prefix + (isTail ? "    " : "│   "), false);
        }
        if (children.size() > 0) {
            children.get(children.size() - 1)
                    .print(prefix + (isTail ?"    " : "│   "), true);
        }
    }

    public void printThis(String prefix, boolean isTail) {
        System.out.println(prefix + (isTail ? "└── " : "├── ") + attribute);
    }
}

class ID3TreeLeaf extends ID3TreeNode {

    private Double attributeValue;
    private Double classValueForAttributeValue;
    public ID3TreeLeaf(ID3TreeNode parent) {
        super(parent);
    }

    public void setClassValueForAttributeValue(Double attrValue, Double classValueForAttributeValue) {
        this.attributeValue = attrValue;
        this.classValueForAttributeValue = classValueForAttributeValue;
    }

    public void printThis(String prefix, boolean isTail) {
        System.out.println(prefix + (isTail ? "└── " : "├── ") + this.attributeValue + " ~> " + this.classValueForAttributeValue);
    }
}